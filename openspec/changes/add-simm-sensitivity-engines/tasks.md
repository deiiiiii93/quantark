# Tasks: Add SIMM Sensitivity Engines

## 1. Module Structure Setup
- [ ] 1.1 Create `simm/sensitivity/__init__.py` with public exports
- [ ] 1.2 Create `simm/sensitivity/base.py` with Sensitivity protocol and base dataclasses

## 2. Sensitivity Data Models
- [ ] 2.1 Add DeltaSensitivity dataclass
- [ ] 2.2 Add VegaSensitivity dataclass
- [ ] 2.3 Add CurvatureSensitivity dataclass
- [ ] 2.4 Add BaseCorrSensitivity dataclass
- [ ] 2.5 Add SensitivityCollection class with grouping methods

## 3. Interest Rate Sensitivity Engine
- [ ] 3.1 Create `simm/sensitivity/ir_engine.py`
- [ ] 3.2 Implement IR delta calculation (PV01 by tenor and sub-curve)
- [ ] 3.3 Implement inflation sensitivity calculation
- [ ] 3.4 Implement cross-currency basis sensitivity calculation
- [ ] 3.5 Implement IR vega calculation (swaption vol sensitivity)
- [ ] 3.6 Implement IR curvature calculation
- [ ] 3.7 Add tenor interpolation for non-standard maturities

## 4. Credit Sensitivity Engine
- [ ] 4.1 Create `simm/sensitivity/credit_engine.py`
- [ ] 4.2 Implement Credit Q delta calculation (CS01 by tenor)
- [ ] 4.3 Implement issuer/seniority classification
- [ ] 4.4 Implement bucket assignment based on credit quality and sector
- [ ] 4.5 Implement base correlation sensitivity (BC01)
- [ ] 4.6 Implement Credit NQ delta calculation
- [ ] 4.7 Implement Credit vega calculation
- [ ] 4.8 Implement Credit curvature calculation

## 5. Equity Sensitivity Engine
- [ ] 5.1 Create `simm/sensitivity/equity_engine.py`
- [ ] 5.2 Implement equity delta calculation using GreeksCalculator
- [ ] 5.3 Implement bucket classification (size, region, sector)
- [ ] 5.4 Add market cap classification (Large >= $2B, Small < $2B)
- [ ] 5.5 Add developed/emerging market classification
- [ ] 5.6 Add sector classification mapping
- [ ] 5.7 Implement index/ETF handling (bucket 11)
- [ ] 5.8 Implement volatility index handling (bucket 12)
- [ ] 5.9 Implement equity vega calculation
- [ ] 5.10 Implement equity curvature calculation with SF(t)

## 6. Commodity Sensitivity Engine
- [ ] 6.1 Create `simm/sensitivity/commodity_engine.py`
- [ ] 6.2 Implement commodity delta calculation
- [ ] 6.3 Implement bucket classification (17 commodity types)
- [ ] 6.4 Implement commodity vega calculation
- [ ] 6.5 Implement commodity curvature calculation

## 7. FX Sensitivity Engine
- [ ] 7.1 Create `simm/sensitivity/fx_engine.py`
- [ ] 7.2 Implement FX delta calculation (1% spot shock)
- [ ] 7.3 Implement FX translation risk from position values
- [ ] 7.4 Implement FX vega calculation (currency pair volatility)
- [ ] 7.5 Implement FX curvature calculation

## 8. Vega Engine (Cross-Risk-Class)
- [ ] 8.1 Create `simm/sensitivity/vega_engine.py`
- [ ] 8.2 Implement vol-weighted vega formula (VR = HVR × σ × vega)
- [ ] 8.3 Implement tenor bucket mapping for option expiries
- [ ] 8.4 Add HVR lookup integration

## 9. Curvature Engine (Cross-Risk-Class)
- [ ] 9.1 Create `simm/sensitivity/curvature_engine.py`
- [ ] 9.2 Implement scaling function SF(t) = 0.5 × min(1, 14/t)
- [ ] 9.3 Implement CVR calculation
- [ ] 9.4 Add curvature for IR risk class (with HVR^-2 scaling)

## 10. Portfolio Adapter
- [ ] 10.1 Create `simm/sensitivity/portfolio_adapter.py`
- [ ] 10.2 Implement PortfolioSensitivityAdapter class
- [ ] 10.3 Add position-to-engine routing logic
- [ ] 10.4 Add batch sensitivity calculation
- [ ] 10.5 Integrate with EquityPortfolio positions
- [ ] 10.6 Integrate with FIPortfolio positions

## 11. CRIF Integration
- [ ] 11.1 Add CRIF-to-Sensitivity conversion in portfolio adapter
- [ ] 11.2 Add Sensitivity-to-CRIF export from calculated sensitivities
- [ ] 11.3 Add sensitivity netting by risk factor

## 12. Bucket Classification Data
- [ ] 12.1 Create equity issuer-to-bucket mapping configuration
- [ ] 12.2 Create commodity-to-bucket mapping configuration
- [ ] 12.3 Create credit issuer-to-bucket mapping configuration
- [ ] 12.4 Add configurable classification overrides

## 13. Testing
- [ ] 13.1 Create `test/test_simm_ir_sensitivity.py`
- [ ] 13.2 Create `test/test_simm_equity_sensitivity.py`
- [ ] 13.3 Create `test/test_simm_credit_sensitivity.py`
- [ ] 13.4 Create `test/test_simm_fx_sensitivity.py`
- [ ] 13.5 Create `test/test_simm_vega_curvature.py`
- [ ] 13.6 Add integration tests with sample portfolios
- [ ] 13.7 Add CRIF round-trip tests

## 14. Documentation
- [ ] 14.1 Add docstrings with SIMM section references
- [ ] 14.2 Document bucket classification rules
- [ ] 14.3 Add example code for sensitivity calculation

