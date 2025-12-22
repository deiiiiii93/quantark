# Product Guidelines: QuantArk

## Tone and Voice
*   **Professional & Academic:** The voice of QuantArk is formal, precise, and rigorous. It reflects the mathematical integrity required for financial engineering. Documentation should avoid slang and prioritize clarity and correctness above all else.

## API Design Principles
*   **Explicit over Implicit:** We favor clarity. Parameter names should be descriptive, and configurations should be explicit. We avoid "magic" behaviors that might obscure the underlying financial logic.
*   **Functional & Immutable:** Pricing engines and calculators should be side-effect free. They take a state (Product and PricingEnvironment) and return a Result, ensuring predictability in complex simulations.
*   **Fluent & Chainable:** While remaining explicit, the API should allow for a readable and logical flow when setting up instruments and market data environments.
*   **Consistency:** Patterns used for Equity derivatives must be mirrored in Fixed Income, FX, and other asset classes to minimize the user's cognitive load.

## Reliability and Safety
*   **Fail-Fast Validation:** The system must raise descriptive exceptions immediately upon encountering invalid inputs, boundary violations, or numerical instability. We do not allow "silent" failures.
*   **Strict Type Safety:** Extensive use of Python type hints and `dataclasses` is required for all public interfaces to ensure data integrity.
*   **Audit-Ready Logging:** The library should provide detailed logging of the pricing process and market data resolution to facilitate troubleshooting and regulatory auditing.

## Visual Identity and Reporting
*   **Professional Minimalism:** Charts and reports use a clean, high-contrast aesthetic suitable for executive summaries and formal risk reporting.
*   **Interactive Exploration:** We utilize dynamic visualizations (e.g., Plotly) to allow users to drill down into risk surfaces and scenario results during exploratory analysis.
*   **Production-Ready Export:** All visual assets must be exportable in high-resolution formats (PDF, PNG) for inclusion in external documents.
*   **Unified Aesthetic:** A consistent color palette and typography must be maintained across all generated plots to reinforce the QuantArk brand.

## Engineering Standards
*   **Test-Driven Excellence:** All new features must be accompanied by comprehensive unit and integration tests, targeting high code coverage (>80%).
*   **Documented Mathematics:** Every pricing algorithm and risk measure must include clear documentation of the underlying formulas and references to standard financial literature.
*   **Clean Code:** We strictly adhere to PEP 8 standards and prioritize long-term maintainability through clear naming conventions and modular design.
