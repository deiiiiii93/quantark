## 1. Implementation

### 1.1 Core Engine Structure
- [ ] 1.1.1 Create `asset/equity/engine/analytical/american_option_engine.py`
- [ ] 1.1.2 Implement `AmericanOptionAnalyticalEngine` class inheriting from `BaseEngine`
- [ ] 1.1.3 Add method selection parameter to `EngineParams` (BS93, BS02, BAW)
- [ ] 1.1.4 Implement `price()` method with routing logic for calls/puts and method selection
- [ ] 1.1.5 Implement input validation method `_validate_inputs()`
- [ ] 1.1.6 Add special case handling (zero dividends, negative rates)

### 1.2 BS93 Method (Bjerksund-Stensland 1993)
- [ ] 1.2.1 Implement `_price_american_call_bs93()` using single-barrier approximation
- [ ] 1.2.2 Implement `_phi_bs93()` auxiliary function
- [ ] 1.2.3 Calculate beta parameter and exercise boundaries (B_0, B_infinity)
- [ ] 1.2.4 Implement trigger price calculation with h_tau
- [ ] 1.2.5 Add put-call transformation for American puts

### 1.3 BS02 Method (Bjerksund-Stensland 2002)
- [ ] 1.3.1 Implement `_price_american_call_bs02()` using two-barrier approximation
- [ ] 1.3.2 Implement `_phi_bs02()` auxiliary function with improved formula
- [ ] 1.3.3 Implement `_psi_bs02()` auxiliary function for second barrier
- [ ] 1.3.4 Implement `_bivariate_normal_cdf()` using scipy.stats.multivariate_normal
- [ ] 1.3.5 Calculate t1 = 0.5(√5 - 1)T and dual trigger prices (I1, I2)
- [ ] 1.3.6 Implement robust alpha calculation with log-space for numerical stability
- [ ] 1.3.7 Add put-call transformation for American puts

### 1.4 BAW Method (Barone-Adesi-Whaley 1987)
- [ ] 1.4.1 Implement `_price_american_call_baw()` using quadratic approximation
- [ ] 1.4.2 Implement `_price_american_put_baw()` with direct put pricing
- [ ] 1.4.3 Calculate M, N, K parameters and q1/q2 seeds
- [ ] 1.4.4 Implement `_find_critical_call_price()` with Newton-Raphson iteration
- [ ] 1.4.5 Implement `_find_critical_put_price()` with Newton-Raphson iteration
- [ ] 1.4.6 Implement `_d1_baw()` helper for d1 calculation
- [ ] 1.4.7 Add discriminant check and European fallback for negative values

### 1.5 Helper Functions
- [ ] 1.5.1 Implement `_european_call_bsm()` for fallback and BAW base calculation
- [ ] 1.5.2 Implement `_european_put_bsm()` for fallback and BAW base calculation
- [ ] 1.5.3 Add numerical stability helpers (`_safe_log`, `_safe_sqrt`, `_safe_exp`)
- [ ] 1.5.4 Add fallback to `BlackScholesEngine` for edge cases
- [ ] 1.5.5 Add parameter clamping (volatility: 0.001-5.0, maturity: 1e-6-30.0)

### 1.6 Documentation and Integration
- [ ] 1.6.1 Add comprehensive docstrings with mathematical formulas for all methods
- [ ] 1.6.2 Document references to original papers (BS93, BS02, BAW87)
- [ ] 1.6.3 Update `asset/equity/engine/analytical/__init__.py` to export new engine

## 2. Testing

### 2.1 Basic Functionality Tests
- [ ] 2.1.1 Create `test/test_american_option_analytical.py`
- [ ] 2.1.2 Test American call pricing with BS93 method
- [ ] 2.1.3 Test American call pricing with BS02 method
- [ ] 2.1.4 Test American call pricing with BAW method
- [ ] 2.1.5 Test American put pricing with BS93 method (via transformation)
- [ ] 2.1.6 Test American put pricing with BS02 method (via transformation)
- [ ] 2.1.7 Test American put pricing with BAW method (direct)

### 2.2 Accuracy and Comparison Tests
- [ ] 2.2.1 Compare BS93 vs BS02 pricing accuracy
- [ ] 2.2.2 Compare BS02 vs BAW pricing accuracy
- [ ] 2.2.3 Verify American prices >= European prices
- [ ] 2.2.4 Verify American prices >= intrinsic value
- [ ] 2.2.5 Test put-call parity relationships (where applicable)

### 2.3 Edge Cases and Special Conditions
- [ ] 2.3.1 Test zero dividend call options (should equal European)
- [ ] 2.3.2 Test negative interest rates
- [ ] 2.3.3 Test near-expiry options (T < 1e-10)
- [ ] 2.3.4 Test deep ITM and deep OTM options
- [ ] 2.3.5 Test extreme volatility (>200%, <1%)
- [ ] 2.3.6 Test very long maturities (>10 years)

### 2.4 Numerical Stability Tests
- [ ] 2.4.1 Test overflow protection in exponential calculations
- [ ] 2.4.2 Test underflow protection in log calculations
- [ ] 2.4.3 Verify no NaN/Inf results for valid inputs
- [ ] 2.4.4 Test fallback to European pricing when needed
- [ ] 2.4.5 Test parameter clamping behavior

### 2.5 Error Handling Tests
- [ ] 2.5.1 Test ValidationError for negative spot/strike/volatility
- [ ] 2.5.2 Test PricingError for unsupported product types
- [ ] 2.5.3 Test NumericalError for computation failures
- [ ] 2.5.4 Test invalid method selection

## 3. Documentation

- [ ] 3.1 Add usage example to `example/american_option_demo.py`
- [ ] 3.2 Include comparison of all three methods in example
- [ ] 3.3 Update README.md to mention American option analytical pricing with three methods