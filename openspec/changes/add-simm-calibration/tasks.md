# Tasks: Add SIMM Calibration Data

## 1. Module Structure Setup
- [ ] 1.1 Create `simm/calibration/__init__.py` with public exports
- [ ] 1.2 Create `simm/calibration/version.py` with SIMM version info

## 2. Interest Rate Parameters
- [ ] 2.1 Create `simm/calibration/ir.py`
- [ ] 2.2 Add IR risk weights by tenor and currency group (36 values)
- [ ] 2.3 Add IR tenor correlation matrix (12x12)
- [ ] 2.4 Add IR sub-curve correlation (99.3%)
- [ ] 2.5 Add IR inflation correlation (24%)
- [ ] 2.6 Add IR cross-currency basis correlation (4%)
- [ ] 2.7 Add IR inter-currency correlation (32%)
- [ ] 2.8 Add IR inflation risk weight (61)
- [ ] 2.9 Add IR cross-currency basis risk weight (21)
- [ ] 2.10 Add IR HVR (0.47) and VRW (0.23)
- [ ] 2.11 Add IR delta concentration thresholds by currency group
- [ ] 2.12 Add IR vega concentration thresholds by currency group

## 3. Credit Qualifying Parameters
- [ ] 3.1 Create `simm/calibration/credit_qualifying.py`
- [ ] 3.2 Add Credit Q risk weights by bucket (12 + residual)
- [ ] 3.3 Add Credit Q intra-bucket correlations (same issuer, different issuer)
- [ ] 3.4 Add Credit Q inter-bucket correlation matrix (12x12)
- [ ] 3.5 Add Credit Q VRW (0.76)
- [ ] 3.6 Add Credit Q base correlation risk weight (10)
- [ ] 3.7 Add Credit Q base correlation inter-index correlation (29%)
- [ ] 3.8 Add Credit Q delta concentration thresholds by bucket
- [ ] 3.9 Add Credit Q vega concentration threshold (360)

## 4. Credit Non-Qualifying Parameters
- [ ] 4.1 Create `simm/calibration/credit_non_qualifying.py`
- [ ] 4.2 Add Credit NQ risk weights by bucket (2 + residual)
- [ ] 4.3 Add Credit NQ intra-bucket correlations
- [ ] 4.4 Add Credit NQ inter-bucket correlation (43%)
- [ ] 4.5 Add Credit NQ VRW (0.76)
- [ ] 4.6 Add Credit NQ delta concentration thresholds
- [ ] 4.7 Add Credit NQ vega concentration threshold (70)

## 5. Equity Parameters
- [ ] 5.1 Create `simm/calibration/equity.py`
- [ ] 5.2 Add Equity risk weights by bucket (12 + residual)
- [ ] 5.3 Add Equity intra-bucket correlations by bucket
- [ ] 5.4 Add Equity inter-bucket correlation matrix (12x12)
- [ ] 5.5 Add Equity HVR (60%)
- [ ] 5.6 Add Equity VRW (0.45, bucket 12: 0.96)
- [ ] 5.7 Add Equity delta concentration thresholds by bucket
- [ ] 5.8 Add Equity vega concentration thresholds by bucket

## 6. Commodity Parameters
- [ ] 6.1 Create `simm/calibration/commodity.py`
- [ ] 6.2 Add Commodity risk weights by bucket (17 buckets)
- [ ] 6.3 Add Commodity intra-bucket correlations by bucket
- [ ] 6.4 Add Commodity inter-bucket correlation matrix (17x17)
- [ ] 6.5 Add Commodity HVR (74%)
- [ ] 6.6 Add Commodity VRW (0.55)
- [ ] 6.7 Add Commodity delta concentration thresholds by bucket
- [ ] 6.8 Add Commodity vega concentration thresholds by bucket

## 7. FX Parameters
- [ ] 7.1 Create `simm/calibration/fx.py`
- [ ] 7.2 Add FX risk weights by volatility group pair (2x2 table)
- [ ] 7.3 Add FX correlations by volatility group (regular calc ccy, high calc ccy)
- [ ] 7.4 Add FX vega/curvature correlation (0.5)
- [ ] 7.5 Add FX HVR (0.57)
- [ ] 7.6 Add FX VRW (0.48)
- [ ] 7.7 Add FX delta concentration thresholds by category
- [ ] 7.8 Add FX vega concentration thresholds by category pair

## 8. Inter-Risk-Class Parameters
- [ ] 8.1 Create `simm/calibration/cross_risk.py`
- [ ] 8.2 Add inter-risk-class correlation matrix (ψ) (6x6)
- [ ] 8.3 Add accessor function for ψ by risk class pair

## 9. Accessor Functions
- [ ] 9.1 Add `get_risk_weight()` unified accessor
- [ ] 9.2 Add `get_intra_bucket_correlation()` accessor
- [ ] 9.3 Add `get_inter_bucket_correlation()` accessor
- [ ] 9.4 Add `get_inter_risk_class_correlation()` accessor
- [ ] 9.5 Add `get_concentration_threshold()` accessor
- [ ] 9.6 Add `get_hvr()` accessor
- [ ] 9.7 Add `get_vrw()` accessor

## 10. Testing
- [ ] 10.1 Create `test/test_simm_calibration_ir.py`
- [ ] 10.2 Create `test/test_simm_calibration_credit.py`
- [ ] 10.3 Create `test/test_simm_calibration_equity.py`
- [ ] 10.4 Create `test/test_simm_calibration_commodity.py`
- [ ] 10.5 Create `test/test_simm_calibration_fx.py`
- [ ] 10.6 Add tests verifying values match ISDA specification
- [ ] 10.7 Add correlation matrix symmetry tests
- [ ] 10.8 Add correlation matrix positive semi-definite tests

## 11. Documentation
- [ ] 11.1 Add docstrings referencing ISDA SIMM v2.6 sections
- [ ] 11.2 Document units for all parameters (bp, %, USD mm)

