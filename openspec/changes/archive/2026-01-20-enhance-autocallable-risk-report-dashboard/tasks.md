## 1. Specification
- [x] 1.1 Add `autocallable-risk-report` spec delta for dashboard enhancements and validate

## 2. Barrier Risk
- [x] 2.1 Add barrier proximity metrics (sigma distance to KI/KO) and report section
- [x] 2.2 Add barrier-zoom grids (KI/KO ±2%) and localized Gamma/Vega surfaces + plots

## 3. Advanced Volatility Risk
- [x] 3.1 Add skew/smile shock model for autocallable report inputs
- [x] 3.2 Add Vanna and Volga surfaces + plots

## 4. Higher-Order Time Greeks
- [x] 4.1 Add Charm (dDelta/dTime) and Color (dGamma/dTime) surfaces

## 5. Dashboard & Lifecycle Context
- [x] 5.1 Add executive dashboard block (vital signs, barrier watch, status indicator)
- [x] 5.2 Add lifecycle context switch (pre-KI vs post-KI focus + recovery probability)

## 6. Scenario Stress & Cashflows
- [x] 6.1 Add stress scenario table (Black Monday, Slow Bleed, custom shocks)
- [x] 6.2 Add conditional cashflow table (expected vs conditional-on-KO date)

## 7. Tests
- [x] 7.1 Add tests for barrier proximity and barrier-zoom grid outputs
- [x] 7.2 Add tests for Vanna/Volga/Charm/Color surface computation
- [x] 7.3 Add report smoke test updates for new sections
