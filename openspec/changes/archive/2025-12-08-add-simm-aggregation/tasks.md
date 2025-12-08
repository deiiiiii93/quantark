# Tasks: Add SIMM Aggregation Engine

## 1. Module Structure Setup
- [x] 1.1 Create `simm/engine/__init__.py` with public exports
- [x] 1.2 Define aggregation result dataclasses

## 2. Concentration Risk Calculator
- [x] 2.1 Create `simm/engine/concentration.py`
- [x] 2.2 Implement IR concentration risk (CR_b = max(1, sqrt(|Σs|/T)))
- [x] 2.3 Implement Credit concentration risk (per issuer/seniority)
- [x] 2.4 Implement Equity/Commodity/FX concentration risk (per risk factor)
- [x] 2.5 Implement Vega concentration risk (VCR)
- [x] 2.6 Implement g_bc factor (min(CR)/max(CR))
- [x] 2.7 Add unit tests for concentration calculations

## 3. Weighted Sensitivity Calculator
- [x] 3.1 Create `simm/engine/weighted_sensitivity.py`
- [x] 3.2 Implement WS = RW × s × CR formula
- [x] 3.3 Handle IR cross-currency basis (no CR scaling)
- [x] 3.4 Handle base correlation (CR = 1)
- [x] 3.5 Add unit tests for weighted sensitivity

## 4. Bucket Aggregator
- [x] 4.1 Create `simm/engine/bucket_aggregator.py`
- [x] 4.2 Implement K_b formula for non-IR risk classes
- [x] 4.3 Implement K formula for IR (currency-level)
- [x] 4.4 Implement f_kl correlation adjustment
- [x] 4.5 Handle residual bucket separately
- [x] 4.6 Add unit tests for bucket aggregation

## 5. Risk Class Aggregator
- [x] 5.1 Create `simm/engine/risk_class_aggregator.py`
- [x] 5.2 Implement DeltaMargin formula (non-IR)
- [x] 5.3 Implement DeltaMargin formula (IR with g_bc)
- [x] 5.4 Implement S_b calculation (capped sum)
- [x] 5.5 Implement VegaMargin formula
- [x] 5.6 Implement CurvatureMargin formula with θ and λ
- [x] 5.7 Implement IR curvature HVR^(-2) scaling
- [x] 5.8 Implement BaseCorrMargin formula
- [x] 5.9 Add residual bucket handling
- [x] 5.10 Add unit tests for risk class aggregation

## 6. Product Class Aggregator
- [x] 6.1 Create `simm/engine/product_class_aggregator.py`
- [x] 6.2 Implement SIMM_product formula
- [x] 6.3 Apply inter-risk-class correlations (ψ matrix)
- [x] 6.4 Add unit tests for product class aggregation

## 7. Add-On Calculator
- [x] 7.1 Create `simm/engine/addon.py`
- [x] 7.2 Implement AddOnFixed calculation
- [x] 7.3 Implement AddOnFactor × Notional calculation
- [x] 7.4 Implement multiplicative scale (MS) application
- [x] 7.5 Add unit tests for add-on calculations

## 8. Main SIMM Calculator
- [x] 8.1 Create `simm/engine/simm_calculator.py`
- [x] 8.2 Implement SIMMCalculator class
- [x] 8.3 Implement sensitivity grouping by product class
- [x] 8.4 Implement risk class margin calculation orchestration
- [x] 8.5 Implement full aggregation pipeline
- [x] 8.6 Implement CRIF input mode
- [x] 8.7 Implement portfolio input mode
- [x] 8.8 Add calculation tracing for debugging

## 9. Vega-Specific Aggregation
- [x] 9.1 Implement Vega concentration risk (VCR)
- [x] 9.2 Implement Vega K_b formula
- [x] 9.3 Implement Vega risk class aggregation with g_bc (IR only)
- [x] 9.4 Apply VRW (vega risk weight)

## 10. Curvature-Specific Aggregation
- [x] 10.1 Implement θ = min(Σ CVR / Σ|CVR|, 0)
- [x] 10.2 Implement λ = (Φ^(-1)(99.5%)² - 1)(1 + θ) - θ
- [x] 10.3 Implement ρ² correlation usage
- [x] 10.4 Implement γ² cross-bucket correlation
- [x] 10.5 Implement residual curvature margin
- [x] 10.6 Add HVR^(-2) scaling for IR curvature

## 11. Integration Testing
- [x] 11.1 Create `test/test_simm_calculator.py`
- [x] 11.2 Add end-to-end tests with sample portfolios
- [x] 11.3 Add regression tests against known SIMM results
- [x] 11.4 Add edge case tests (empty buckets, single sensitivities)
- [x] 11.5 Add numerical precision tests

## 12. Performance Optimization
- [x] 12.1 Use NumPy for matrix operations
- [x] 12.2 Optimize correlation lookups
- [x] 12.3 Add caching for repeated calculations
- [x] 12.4 Profile and optimize hot paths

## 13. Documentation
- [x] 13.1 Add docstrings with SIMM formula references
- [x] 13.2 Document calculation flow
- [x] 13.3 Add example usage code

