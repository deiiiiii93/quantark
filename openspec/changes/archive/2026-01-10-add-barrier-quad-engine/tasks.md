## 1. Core Engine Implementation

- [x] 1.1 Create `asset/equity/engine/quad/barrier_quad_engine.py`
- [x] 1.2 Implement `BarrierQuadEngine` class inheriting from `BaseEngine`
- [x] 1.3 Set `engine_type = EngineType.QUADRATURE`
- [x] 1.4 Implement `price()` method for `BarrierOption`

## 2. FFT Convolution Implementation

- [x] 2.1 Implement log-price grid setup with barrier-aware bounds
- [x] 2.2 Implement transition kernel `omega(x)` function
- [x] 2.3 Implement Simpson's rule integration with boundary corrections
- [x] 2.4 Implement FFT-based convolution for backward recursion
- [x] 2.5 Implement factor-based payoff decomposition (asset1-3, cash1-3)

## 3. Barrier Type Support

- [x] 3.1 Support UP_OUT barrier (knock-out when price goes up)
- [x] 3.2 Support DOWN_OUT barrier (knock-out when price goes down)
- [x] 3.3 Support UP_IN barrier (knock-in when price goes up)
- [x] 3.4 Support DOWN_IN barrier (knock-in when price goes down)
- [x] 3.5 Implement knock-in via KI = Vanilla - KO identity

## 4. Observation Schedule Support

- [x] 4.1 Support discrete observation dates from `ObservationSchedule`
- [x] 4.2 Support legacy `observation_dates` list
- [x] 4.3 Handle barrier checking only at observation times
- [x] 4.4 Support time-varying barriers from schedule

## 5. Rebate Handling

- [x] 5.1 Support rebate payment at barrier hit
- [x] 5.2 Support rebate payment at expiry
- [x] 5.3 Discount rebates appropriately based on payment timing

## 6. Integration

- [x] 6.1 Update `asset/equity/engine/quad/__init__.py` to export `BarrierQuadEngine`
- [x] 6.2 Update `asset/equity/engine/__init__.py` to export `BarrierQuadEngine`
- [x] 6.3 Ensure compatibility with existing `QuadParams`

## 7. Testing

- [x] 7.1 Create `test/test_barrier_quad_engine.py`
- [x] 7.2 Test all 4 barrier types vs analytical engine
- [x] 7.3 Test discrete observation schedules
- [x] 7.4 Test rebate handling
- [x] 7.5 Test convergence with grid size
- [x] 7.6 Test edge cases (barrier at spot, near expiry)

## 8. Documentation

- [x] 8.1 Create `example/barrier_quad_demo.py`
- [x] 8.2 Add docstrings to all public methods
