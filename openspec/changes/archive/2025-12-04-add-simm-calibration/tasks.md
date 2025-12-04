# Tasks: Add SIMM Calibration Data

## 1. Module Structure Setup
- [x] 1.1 Create `simm/calibration/__init__.py` with public exports
- [x] 1.2 Create `simm/calibration/version.py` with SIMM version info

## 2. Interest Rate Parameters
- [x] 2.1 Create `simm/calibration/ir.py`
- [x] 2.2 Add IR risk weights by tenor and currency group (36 values)
- [x] 2.3 Add IR tenor correlation matrix (12x12)
- [x] 2.4 Add IR sub-curve correlation (99.3%)
- [x] 2.5 Add IR inflation correlation (24%)
- [x] 2.6 Add IR cross-currency basis correlation (4%)
- [x] 2.7 Add IR inter-currency correlation (32%)
- [x] 2.8 Add IR inflation risk weight (61)
- [x] 2.9 Add IR cross-currency basis risk weight (21)
- [x] 2.10 Add IR HVR (0.47) and VRW (0.23)
- [x] 2.11 Add IR delta concentration thresholds by currency group
- [x] 2.12 Add IR vega concentration thresholds by currency group

## 3. Credit Qualifying Parameters
- [x] 3.1 Create `simm/calibration/credit_qualifying.py`
- [x] 3.2 Add Credit Q risk weights by bucket (12 + residual)
- [x] 3.3 Add Credit Q intra-bucket correlations (same issuer, different issuer)
- [x] 3.4 Add Credit Q inter-bucket correlation matrix (12x12)
- [x] 3.5 Add Credit Q VRW (0.76)
- [x] 3.6 Add Credit Q base correlation risk weight (10)
- [x] 3.7 Add Credit Q base correlation inter-index correlation (29%)
- [x] 3.8 Add Credit Q delta concentration thresholds by bucket
- [x] 3.9 Add Credit Q vega concentration threshold (360)

## 4. Credit Non-Qualifying Parameters
- [x] 4.1 Create `simm/calibration/credit_non_qualifying.py`
- [x] 4.2 Add Credit NQ risk weights by bucket (2 + residual)
- [x] 4.3 Add Credit NQ intra-bucket correlations
- [x] 4.4 Add Credit NQ inter-bucket correlation (43%)
- [x] 4.5 Add Credit NQ VRW (0.76)
- [x] 4.6 Add Credit NQ delta concentration thresholds
- [x] 4.7 Add Credit NQ vega concentration threshold (70)

## 5. Equity Parameters
- [x] 5.1 Create `simm/calibration/equity.py`
- [x] 5.2 Add Equity risk weights by bucket (12 + residual)
- [x] 5.3 Add Equity intra-bucket correlations by bucket
- [x] 5.4 Add Equity inter-bucket correlation matrix (12x12)
- [x] 5.5 Add Equity HVR (60%)
- [x] 5.6 Add Equity VRW (0.45, bucket 12: 0.96)
- [x] 5.7 Add Equity delta concentration thresholds by bucket
- [x] 5.8 Add Equity vega concentration thresholds by bucket

## 6. Commodity Parameters
- [x] 6.1 Create `simm/calibration/commodity.py`
- [x] 6.2 Add Commodity risk weights by bucket (17 buckets)
- [x] 6.3 Add Commodity intra-bucket correlations by bucket
- [x] 6.4 Add Commodity inter-bucket correlation matrix (17x17)
- [x] 6.5 Add Commodity HVR (74%)
- [x] 6.6 Add Commodity VRW (0.55)
- [x] 6.7 Add Commodity delta concentration thresholds by bucket
- [x] 6.8 Add Commodity vega concentration thresholds by bucket

## 7. FX Parameters
- [x] 7.1 Create `simm/calibration/fx.py`
- [x] 7.2 Add FX risk weights by volatility group pair (2x2 table)
- [x] 7.3 Add FX correlations by volatility group (regular calc ccy, high calc ccy)
- [x] 7.4 Add FX vega/curvature correlation (0.5)
- [x] 7.5 Add FX HVR (0.57)
- [x] 7.6 Add FX VRW (0.48)
- [x] 7.7 Add FX delta concentration thresholds by category
- [x] 7.8 Add FX vega concentration thresholds by category pair

## 8. Inter-Risk-Class Parameters
- [x] 8.1 Create `simm/calibration/cross_risk.py`
- [x] 8.2 Add inter-risk-class correlation matrix (ψ) (6x6)
- [x] 8.3 Add accessor function for ψ by risk class pair

## 9. Accessor Functions
- [x] 9.1 Add `get_risk_weight()` unified accessor
- [x] 9.2 Add `get_intra_bucket_correlation()` accessor
- [x] 9.3 Add `get_inter_bucket_correlation()` accessor
- [x] 9.4 Add `get_inter_risk_class_correlation()` accessor
- [x] 9.5 Add `get_concentration_threshold()` accessor
- [x] 9.6 Add `get_hvr()` accessor
- [x] 9.7 Add `get_vrw()` accessor

## 10. Testing
- [x] 10.1 Create `test/test_simm_calibration_ir.py`
- [x] 10.2 Create `test/test_simm_calibration_credit.py`
- [x] 10.3 Create `test/test_simm_calibration_equity.py`
- [x] 10.4 Create `test/test_simm_calibration_commodity.py`
- [x] 10.5 Create `test/test_simm_calibration_fx.py`
- [x] 10.6 Add tests verifying values match ISDA specification
- [x] 10.7 Add correlation matrix symmetry tests
- [x] 10.8 Add correlation matrix positive semi-definite tests

## 11. Documentation
- [x] 11.1 Add docstrings referencing ISDA SIMM v2.6 sections
- [x] 11.2 Document units for all parameters (bp, %, USD mm)

