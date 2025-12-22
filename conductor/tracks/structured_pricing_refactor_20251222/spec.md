# Specification: Consolidate and Refactor Structured Product Pricing

## Goal
Standardize the pricing interface and shared logic for structured products (Snowballs, Airbags) to ensure consistency across analytical, Monte Carlo, and PDE engines.

## Context
Currently, structured products like Snowballs and Airbags share similar logic (KO/KI barriers, complex maturity payoffs) but are implemented with some duplication and varying interfaces. This refactor will create a shared base and consistent data flow.

## Scope
- New base class for structured products.
- Refactored `SnowballOption`.
- Updated pricing engines to use the standardized interface.
- Enhanced test suite for all pricing methods.
