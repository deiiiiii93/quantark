# Tasks: Add SIMM Aggregation Engine

## 1. Module Structure Setup
- [ ] 1.1 Create `simm/engine/__init__.py` with public exports
- [ ] 1.2 Define aggregation result dataclasses

## 2. Concentration Risk Calculator
- [ ] 2.1 Create `simm/engine/concentration.py`
- [ ] 2.2 Implement IR concentration risk (CR_b = max(1, sqrt(|Σs|/T)))
- [ ] 2.3 Implement Credit concentration risk (per issuer/seniority)
- [ ] 2.4 Implement Equity/Commodity/FX concentration risk (per risk factor)
- [ ] 2.5 Implement Vega concentration risk (VCR)
- [ ] 2.6 Implement g_bc factor (min(CR)/max(CR))
- [ ] 2.7 Add unit tests for concentration calculations

## 3. Weighted Sensitivity Calculator
- [ ] 3.1 Create `simm/engine/weighted_sensitivity.py`
- [ ] 3.2 Implement WS = RW × s × CR formula
- [ ] 3.3 Handle IR cross-currency basis (no CR scaling)
- [ ] 3.4 Handle base correlation (CR = 1)
- [ ] 3.5 Add unit tests for weighted sensitivity

## 4. Bucket Aggregator
- [ ] 4.1 Create `simm/engine/bucket_aggregator.py`
- [ ] 4.2 Implement K_b formula for non-IR risk classes
- [ ] 4.3 Implement K formula for IR (currency-level)
- [ ] 4.4 Implement f_kl correlation adjustment
- [ ] 4.5 Handle residual bucket separately
- [ ] 4.6 Add unit tests for bucket aggregation

## 5. Risk Class Aggregator
- [ ] 5.1 Create `simm/engine/risk_class_aggregator.py`
- [ ] 5.2 Implement DeltaMargin formula (non-IR)
- [ ] 5.3 Implement DeltaMargin formula (IR with g_bc)
- [ ] 5.4 Implement S_b calculation (capped sum)
- [ ] 5.5 Implement VegaMargin formula
- [ ] 5.6 Implement CurvatureMargin formula with θ and λ
- [ ] 5.7 Implement IR curvature HVR^(-2) scaling
- [ ] 5.8 Implement BaseCorrMargin formula
- [ ] 5.9 Add residual bucket handling
- [ ] 5.10 Add unit tests for risk class aggregation

## 6. Product Class Aggregator
- [ ] 6.1 Create `simm/engine/product_class_aggregator.py`
- [ ] 6.2 Implement SIMM_product formula
- [ ] 6.3 Apply inter-risk-class correlations (ψ matrix)
- [ ] 6.4 Add unit tests for product class aggregation

## 7. Add-On Calculator
- [ ] 7.1 Create `simm/engine/addon.py`
- [ ] 7.2 Implement AddOnFixed calculation
- [ ] 7.3 Implement AddOnFactor × Notional calculation
- [ ] 7.4 Implement multiplicative scale (MS) application
- [ ] 7.5 Add unit tests for add-on calculations

## 8. Main SIMM Calculator
- [ ] 8.1 Create `simm/engine/simm_calculator.py`
- [ ] 8.2 Implement SIMMCalculator class
- [ ] 8.3 Implement sensitivity grouping by product class
- [ ] 8.4 Implement risk class margin calculation orchestration
- [ ] 8.5 Implement full aggregation pipeline
- [ ] 8.6 Implement CRIF input mode
- [ ] 8.7 Implement portfolio input mode
- [ ] 8.8 Add calculation tracing for debugging

## 9. Vega-Specific Aggregation
- [ ] 9.1 Implement Vega concentration risk (VCR)
- [ ] 9.2 Implement Vega K_b formula
- [ ] 9.3 Implement Vega risk class aggregation with g_bc (IR only)
- [ ] 9.4 Apply VRW (vega risk weight)

## 10. Curvature-Specific Aggregation
- [ ] 10.1 Implement θ = min(Σ CVR / Σ|CVR|, 0)
- [ ] 10.2 Implement λ = (Φ^(-1)(99.5%)² - 1)(1 + θ) - θ
- [ ] 10.3 Implement ρ² correlation usage
- [ ] 10.4 Implement γ² cross-bucket correlation
- [ ] 10.5 Implement residual curvature margin
- [ ] 10.6 Add HVR^(-2) scaling for IR curvature

## 11. Integration Testing
- [ ] 11.1 Create `test/test_simm_calculator.py`
- [ ] 11.2 Add end-to-end tests with sample portfolios
- [ ] 11.3 Add regression tests against known SIMM results
- [ ] 11.4 Add edge case tests (empty buckets, single sensitivities)
- [ ] 11.5 Add numerical precision tests

## 12. Performance Optimization
- [ ] 12.1 Use NumPy for matrix operations
- [ ] 12.2 Optimize correlation lookups
- [ ] 12.3 Add caching for repeated calculations
- [ ] 12.4 Profile and optimize hot paths

## 13. Documentation
- [ ] 13.1 Add docstrings with SIMM formula references
- [ ] 13.2 Document calculation flow
- [ ] 13.3 Add example usage code

