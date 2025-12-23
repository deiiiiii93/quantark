# Specification: Asian Observation Record Examples

## Overview
This track adds comprehensive example cases to `example/asian_option_demo.py` to demonstrate the initialization and usage of the `AsianOption` product using the `AsianObservationRecord` structure. Currently, the demo primarily uses legacy list-based observation schedules.

## Functional Requirements
- Implement a new demonstration function `demonstrate_observation_records()` in `example/asian_option_demo.py`.
- **Scenario A: Historical Observations**: Show how to initialize an `AsianOption` with past observations that have fixed prices, demonstrating the impact on the current average.
- **Scenario B: Future Observations**: Demonstrate setting up future observation points using records without observed prices.
- **Scenario C: Mixed History (Mid-life Option)**: Demonstrate a "mid-life" option scenario where some observations are in the past (with prices) and some are in the future (relative to a valuation date).
- **Scenario D: Date-based Resolution**: Demonstrate the use of `observation_date` within `AsianObservationRecord` and how it resolves to times using a `PricingEnvironment`.

## Tech Stack & Dependencies
- **Product**: `asset.equity.product.option.AsianOption` and `AsianObservationRecord`.
- **Environment**: `priceenv.PricingEnvironment` for date-to-time resolution.
- **Utilities**: `util.enum` for `OptionType`, `AveragingType`, etc.

## Acceptance Criteria
- `example/asian_option_demo.py` executes without errors.
- The output clearly prints the results of the new `AsianObservationRecord` scenarios.
- The `demonstrate_observation_records()` function is integrated into the `main()` execution flow of the demo.

## Out of Scope
- Modifying the core logic of `AsianOption.py` or `AsianObservationRecord`.
- Adding new pricing engines or risk measures.
