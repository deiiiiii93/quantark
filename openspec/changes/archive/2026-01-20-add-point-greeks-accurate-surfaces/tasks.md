## 1. Specification
- [x] 1.1 Add `greeks-calculator` and `autocallable-risk-report` spec deltas and validate

## 2. Point Greeks
- [x] 2.1 Add Vanna/Volga and dDelta/dq calculations in `GreeksCalculator`
- [x] 2.2 Define bump conventions and expose outputs in result dict
- [x] 2.3 Add greeks enums (common + equity) and accept enums/strings in API

## 3. High-Accuracy Surfaces
- [x] 3.1 Add optional report flag to compute surfaces via per-node GreeksCalculator
- [x] 3.2 Use point Greeks for executive dashboard when enabled

## 4. Tests
- [x] 4.1 Add unit tests for new point Greeks
- [x] 4.2 Add report smoke test for high-accuracy mode
