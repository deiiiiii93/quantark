# Tasks: Add Convertible Bond Risk Metrics

## 1. Floor Bond Implementation
- [x] 1.1 Add `calculate_floor_bond_price()` method to `ConvertibleBondEngine`
- [x] 1.2 Implement risky discounting using risk-free rate + credit spread
- [x] 1.3 Add `floor_bond_dv01()` method for floor bond DV01
- [x] 1.4 Add `floor_bond_cs01()` method for floor bond CS01
- [x] 1.5 Add `floor_bond_duration()` method for floor bond modified duration
- [x] 1.6 Add `floor_bond_convexity()` method for floor bond convexity
- [x] 1.7 Write unit tests for floor bond methods

## 2. Convertible Bond Interest Rate and Credit Risk Metrics
- [x] 2.1 Add `dv01()` method using numerical rate bump
- [x] 2.2 Add `cs01()` method using numerical credit spread bump
- [x] 2.3 Add `modified_duration()` method derived from DV01
- [x] 2.4 Add `convexity()` method using central difference
- [x] 2.5 Write unit tests for convertible risk metrics

## 3. Result Container Extension
- [x] 3.1 Extend `ConvertibleBondResult` with new fields
- [x] 3.2 Update `price_with_details()` to populate new fields
- [x] 3.3 Write tests for extended result container

## 4. Documentation and Examples
- [x] 4.1 Update `example/convertible_bond_demo.py` with risk metrics demo
- [x] 4.2 Add docstrings to all new methods

## Dependencies
- Tasks 1.x can be done in parallel with 2.x
- Task 3.x depends on 1.x and 2.x completion
- Task 4.x depends on all prior tasks

## Parallelizable Work
- 1.1-1.5 (Floor bond methods) and 2.1-2.3 (Convertible metrics) can be developed in parallel
