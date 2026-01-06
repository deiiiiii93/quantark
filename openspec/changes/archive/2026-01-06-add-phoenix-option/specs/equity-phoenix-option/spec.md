## ADDED Requirements

### Requirement: Phoenix Option Product

The system SHALL provide a `PhoenixOption` class that represents an autocallable structured product with periodic coupon payments when spot exceeds a coupon barrier at observation dates.

#### Scenario: Create standard Phoenix option
- **GIVEN** initial_price=100, strike=100, maturity=1.0, notional=1_000_000
- **AND** barrier_config with ko_barrier=103, ki_barrier=75, ko_rate=0.15
- **AND** coupon_config with coupon_barrier=85, coupon_rate=0.01, memory_coupon=True
- **WHEN** PhoenixOption is created with these parameters
- **THEN** a valid PhoenixOption instance is returned
- **AND** is_reverse=False by default (standard direction)
- **AND** option_type=PUT (short put exposure on KI)

#### Scenario: Create reverse Phoenix option
- **GIVEN** initial_price=100, strike=100, maturity=1.0, is_reverse=True
- **AND** barrier_config with ko_barrier=97 (down barrier), ki_barrier=125 (up barrier)
- **AND** coupon_config with coupon_barrier=115
- **WHEN** PhoenixOption is created with is_reverse=True
- **THEN** option_type=CALL (short call exposure on KI)
- **AND** KO triggers when spot <= ko_barrier
- **AND** KI triggers when spot >= ki_barrier

#### Scenario: Validation error for invalid coupon barrier
- **GIVEN** coupon_barrier <= 0
- **WHEN** PhoenixOption is created
- **THEN** ValidationError is raised with message containing "coupon_barrier"

---

### Requirement: Coupon Barrier Configuration

The system SHALL provide a `CouponBarrierConfig` dataclass for configuring coupon barrier settings including day count convention and memory coupon feature.

#### Scenario: Create coupon config with ACT/365 day count
- **GIVEN** coupon_barrier=85, coupon_rate=0.01
- **AND** day_count_convention=DayCountConvention.ACT_365
- **WHEN** CouponBarrierConfig is created
- **THEN** coupon year fractions are calculated using ACT/365 convention

#### Scenario: Create coupon config with 30/360 US day count
- **GIVEN** coupon_barrier=85, coupon_rate=0.01
- **AND** day_count_convention=DayCountConvention.THIRTY_360_US
- **WHEN** CouponBarrierConfig is created
- **THEN** coupon year fractions are calculated using 30/360 US convention

#### Scenario: Coupon pay type INSTANT
- **GIVEN** coupon_pay_type=CouponPayType.INSTANT
- **WHEN** coupon barrier is hit at observation
- **THEN** coupon is paid immediately at observation date

#### Scenario: Coupon pay type EXPIRY
- **GIVEN** coupon_pay_type=CouponPayType.EXPIRY
- **WHEN** coupon barrier is hit at observations
- **THEN** coupons are accumulated and paid at maturity

#### Scenario: Memory coupon enabled
- **GIVEN** memory_coupon=True
- **AND** coupon barrier not hit at observation 1
- **AND** coupon barrier hit at observation 2
- **WHEN** coupon is calculated at observation 2
- **THEN** coupon includes accumulated amount from observation 1

#### Scenario: Memory coupon disabled
- **GIVEN** memory_coupon=False
- **AND** coupon barrier not hit at observation 1
- **AND** coupon barrier hit at observation 2
- **WHEN** coupon is calculated at observation 2
- **THEN** coupon includes only observation 2 amount (no accumulation)

---

### Requirement: Coupon Barrier Triggering

The system SHALL provide methods to check if coupon barrier is triggered at each observation date.

#### Scenario: Coupon triggered when spot above barrier (standard)
- **GIVEN** a standard PhoenixOption with coupon_barrier=85
- **AND** is_reverse=False
- **WHEN** is_coupon_triggered(spot=90, obs_idx=0) is called
- **THEN** True is returned (spot >= coupon_barrier)

#### Scenario: Coupon not triggered when spot below barrier (standard)
- **GIVEN** a standard PhoenixOption with coupon_barrier=85
- **AND** is_reverse=False
- **WHEN** is_coupon_triggered(spot=80, obs_idx=0) is called
- **THEN** False is returned (spot < coupon_barrier)

#### Scenario: Coupon triggered when spot below barrier (reverse)
- **GIVEN** a reverse PhoenixOption with coupon_barrier=115
- **AND** is_reverse=True
- **WHEN** is_coupon_triggered(spot=110, obs_idx=0) is called
- **THEN** True is returned (spot <= coupon_barrier)

#### Scenario: Time-varying coupon barrier
- **GIVEN** coupon_barrier=[85, 84, 83, 82] (decreasing)
- **WHEN** is_coupon_triggered(spot=83.5, obs_idx=2) is called
- **THEN** True is returned (spot >= coupon_barrier[2]=83)

---

### Requirement: Coupon Payoff Calculation

The system SHALL calculate coupon payoffs using day count conventions for year fraction calculation.

#### Scenario: Calculate coupon with ACT/365
- **GIVEN** notional=1_000_000, coupon_rate=0.12 (12% annual)
- **AND** day_count_convention=DayCountConvention.ACT_365
- **AND** accrual period of 91 days
- **WHEN** get_coupon_payoff(obs_idx) is called
- **THEN** coupon = notional * coupon_rate * (91/365) = 29,917.81

#### Scenario: Calculate coupon with 30/360 US
- **GIVEN** notional=1_000_000, coupon_rate=0.12 (12% annual)
- **AND** day_count_convention=DayCountConvention.THIRTY_360_US
- **AND** accrual period of 3 months
- **WHEN** get_coupon_payoff(obs_idx) is called
- **THEN** coupon = notional * coupon_rate * (90/360) = 30,000.00

#### Scenario: Memory coupon accumulation
- **GIVEN** memory_coupon=True, coupon_rate=0.01 per period
- **AND** observations at [0.25, 0.5, 0.75, 1.0]
- **AND** coupon triggered at obs 0 (0.25), missed at obs 1 (0.5), triggered at obs 2 (0.75)
- **WHEN** get_accumulated_coupon() is called at obs 2
- **THEN** accumulated = coupon(0.25) + coupon(0.5) + coupon(0.75)

---

### Requirement: Knock-Out Barrier Triggering

The system SHALL provide methods to check if KO barrier is triggered, following the same logic as SnowballOption.

#### Scenario: KO triggered (standard)
- **GIVEN** a standard PhoenixOption with ko_barrier=103
- **WHEN** is_ko_triggered(spot=105, obs_idx=0) is called
- **THEN** True is returned (spot >= ko_barrier)

#### Scenario: KO triggered (reverse)
- **GIVEN** a reverse PhoenixOption with ko_barrier=97
- **WHEN** is_ko_triggered(spot=95, obs_idx=0) is called
- **THEN** True is returned (spot <= ko_barrier)

#### Scenario: Step-down KO barrier
- **GIVEN** ko_barrier=[103, 102, 101, 100] (decreasing)
- **WHEN** is_ko_triggered(spot=101.5, obs_idx=2) is called
- **THEN** True is returned (spot >= ko_barrier[2]=101)

---

### Requirement: Knock-In Barrier Triggering

The system SHALL provide methods to check if KI barrier is triggered, following the same logic as SnowballOption.

#### Scenario: KI triggered (standard)
- **GIVEN** a standard PhoenixOption with ki_barrier=75
- **WHEN** is_ki_triggered(spot=70, obs_idx=0) is called
- **THEN** True is returned (spot <= ki_barrier)

#### Scenario: KI triggered (reverse)
- **GIVEN** a reverse PhoenixOption with ki_barrier=125
- **WHEN** is_ki_triggered(spot=130, obs_idx=0) is called
- **THEN** True is returned (spot >= ki_barrier)

---

### Requirement: Maturity Payoff V0 (Not Knocked-In)

The system SHALL calculate maturity payoff for V0 state (not knocked-in, no KO) including accumulated coupons.

#### Scenario: V0 payoff with accumulated coupons
- **GIVEN** a PhoenixOption with notional=1_000_000, rebate_rate=0.05
- **AND** accumulated_coupons=30_000
- **WHEN** get_maturity_payoff_v0(spot, accumulated_coupons) is called
- **THEN** payoff = principal + rebate + accumulated_coupons

#### Scenario: V0 payoff with call rebate
- **GIVEN** call_rebate_enabled=True, call_strike=105
- **AND** spot=110 at maturity
- **WHEN** get_maturity_payoff_v0(spot, accumulated_coupons) is called
- **THEN** payoff includes call-style rebate on (spot - call_strike)

---

### Requirement: Maturity Payoff V1 (Knocked-In)

The system SHALL calculate maturity payoff for V1 state (knocked-in, no KO) with participation in downside.

#### Scenario: V1 payoff with participation
- **GIVEN** a standard PhoenixOption with participation_rate=1.0, strike=100
- **AND** spot=80 at maturity (KI triggered earlier)
- **WHEN** get_maturity_payoff_v1(spot) is called
- **THEN** payoff = principal - participation_rate * max(strike - spot, 0)
- **AND** payoff = 1_000_000 - 1.0 * 200_000 = 800_000

#### Scenario: V1 payoff with protection
- **GIVEN** protection_type=ProtectionType.PARTIAL, protection_rate=0.8
- **AND** spot=80 at maturity
- **WHEN** get_maturity_payoff_v1(spot) is called
- **THEN** loss is floored at protection_rate * initial_price

---

### Requirement: KO Payoff

The system SHALL calculate KO payoff including accumulated coupons and current period coupon (if earned).

#### Scenario: KO payoff with accumulated coupons
- **GIVEN** a PhoenixOption with ko_rate=0.15, notional=1_000_000
- **AND** KO triggered at obs_idx=2 (0.5 years)
- **AND** accumulated_coupons=20_000
- **WHEN** get_ko_payoff(spot, obs_idx, accumulated_coupons) is called
- **THEN** payoff = principal + ko_coupon + accumulated_coupons

#### Scenario: KO payoff includes current coupon if earned
- **GIVEN** coupon barrier also hit at KO observation
- **WHEN** get_ko_payoff() is called
- **THEN** current period coupon is included in accumulated_coupons

---

### Requirement: Phoenix Option Helpers

The system SHALL provide factory functions for creating common Phoenix option variants.

#### Scenario: Create standard Phoenix with defaults
- **GIVEN** initial_price=100, strike=100, maturity=1.0
- **WHEN** create_standard_phoenix(initial_price, strike, maturity) is called
- **THEN** PhoenixOption is returned with:
  - ko_barrier = 103 (103% of initial_price)
  - ki_barrier = 75 (75% of initial_price)
  - coupon_barrier = 85 (85% of initial_price)
  - coupon_rate = 0.01 (1% per period)
  - memory_coupon = True
  - 12 monthly observations

#### Scenario: Create step-down Phoenix
- **GIVEN** initial_price=100, strike=100, maturity=1.0, stepdown_rate=0.005
- **WHEN** create_stepdown_phoenix(initial_price, strike, maturity, stepdown_rate) is called
- **THEN** PhoenixOption is returned with:
  - ko_barrier = [103.0, 102.5, 102.0, ...] (decreasing)
  - coupon_barrier = [85.0, 84.5, 84.0, ...] (decreasing)

#### Scenario: Create reverse Phoenix
- **GIVEN** initial_price=100, strike=100, maturity=1.0
- **WHEN** create_reverse_phoenix(initial_price, strike, maturity) is called
- **THEN** PhoenixOption is returned with:
  - is_reverse = True
  - ko_barrier = 97 (down barrier)
  - ki_barrier = 125 (up barrier)
  - coupon_barrier = 115

---

### Requirement: Module Exports

The Phoenix option classes and helpers SHALL be exported from the equity option module.

#### Scenario: Import PhoenixOption from option module
- **GIVEN** the asset.equity.product.option module
- **WHEN** `from asset.equity.product.option import PhoenixOption` is executed
- **THEN** PhoenixOption class is accessible

#### Scenario: Import CouponBarrierConfig from option module
- **GIVEN** the asset.equity.product.option module
- **WHEN** `from asset.equity.product.option import CouponBarrierConfig` is executed
- **THEN** CouponBarrierConfig class is accessible

#### Scenario: Import helpers from phoenix_helpers module
- **GIVEN** the asset.equity.product.option.phoenix_helpers module
- **WHEN** importing create_standard_phoenix, create_stepdown_phoenix
- **THEN** all helper functions are accessible

---

### Requirement: Input Validation

All Phoenix option constructors and methods MUST validate input parameters.

#### Scenario: Reject negative initial price
- **GIVEN** initial_price=-100
- **WHEN** PhoenixOption is created
- **THEN** ValidationError is raised with message containing "initial_price"

#### Scenario: Reject zero maturity
- **GIVEN** maturity=0
- **WHEN** PhoenixOption is created
- **THEN** ValidationError is raised with message containing "maturity"

#### Scenario: Reject negative coupon rate
- **GIVEN** coupon_rate=-0.01
- **WHEN** CouponBarrierConfig is created
- **THEN** ValidationError is raised with message containing "coupon_rate"

#### Scenario: Reject invalid day count convention
- **GIVEN** day_count_convention is not a DayCountConvention enum
- **WHEN** CouponBarrierConfig is created
- **THEN** ValidationError is raised with message containing "day_count_convention"
