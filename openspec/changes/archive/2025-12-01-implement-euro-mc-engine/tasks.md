# Implementation Tasks

## 1. Implementation
- [x] 1.1 Study `GBMPathGenerator` interface and usage patterns
- [x] 1.2 Study `run_rqmc` driver interface
- [x] 1.3 Implement `EuropeanMCEngine` class with base structure
- [x] 1.4 Implement normal MC pricing method
- [x] 1.5 Implement QMC pricing method (Sobol)
- [x] 1.6 Implement RQMC pricing method with adaptive batching
- [x] 1.7 Add proper error handling and validation
- [x] 1.8 Implement standard error estimation
- [x] 1.9 Add docstrings and type hints

## 2. Testing
- [x] 2.1 Write unit tests for each MC method
- [x] 2.2 Validate against Black-Scholes analytical prices
- [x] 2.3 Test convergence properties (MC vs QMC vs RQMC)
- [x] 2.4 Test variance reduction effectiveness
- [x] 2.5 Test edge cases (near expiry, extreme strikes, etc.)

## 3. Documentation
- [x] 3.1 Add usage examples in docstrings
- [x] 3.2 Document method selection trade-offs
- [x] 3.3 Create comprehensive demo in `example/european_mc_demo.py`