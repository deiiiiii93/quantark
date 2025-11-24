"""
Standalone test for bond implementation without importing asset.
"""
import sys
sys.path.insert(0, '/Users/fuxinyao/QuantArk')

from datetime import datetime
import math

# Direct imports to avoid asset module
from asset.bond.product.couponbond.fixed_bond import create_simple_fixed_bond
from asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from param.rrf.rate_curve import FlatRateCurve, LinearRateCurve
from priceenv.pricing_environment import PricingEnvironment
from util.enum.bond_enums import PaymentFrequency
from util.calendar.day_counter import DayCountConvention

print("=" * 80)
print("Testing Bond Implementation")
print("=" * 80)

# Test 1: Create a simple bond
print("\n1. Creating a simple 5-year bond with 5% coupon...")
bond = create_simple_fixed_bond(
    issue_date=datetime(2023, 1, 1),
    maturity_date=datetime(2028, 1, 1),
    notional=1000.0,
    coupon_rate=0.05,
    payment_frequency=PaymentFrequency.SEMI_ANNUAL,
    day_count_convention=DayCountConvention.ACT_ACT_ISDA
)
print(f"   ✓ Bond created: {bond}")
print(f"   ✓ Regular coupon payment: ${bond.get_coupon_payment():.2f}")

# Test 2: Generate cashflows
print("\n2. Generating cashflow schedule...")
cashflows = bond.get_all_cashflows()
print(f"   ✓ Generated {len(cashflows)} cashflows")
print(f"   ✓ First payment date: {cashflows[0].payment_date.date()}")
print(f"   ✓ Last payment date: {cashflows[-1].payment_date.date()}")
print(f"   ✓ Last payment amount (principal + coupon): ${cashflows[-1].amount:.2f}")

# Test 3: Price the bond
print("\n3. Pricing the bond...")
valuation_date = datetime(2024, 1, 1)
rate_curve = FlatRateCurve(rate=0.04)
pricing_env = PricingEnvironment(
    rate_curve=rate_curve,
    valuation_date=valuation_date
)
engine = BondDiscountEngine(pricing_env)

dirty_price = engine.dirty_price(bond)
clean_price = engine.clean_price(bond)
accrued = engine.accrued_interest(bond)

print(f"   ✓ Dirty price: ${dirty_price:.2f}")
print(f"   ✓ Clean price: ${clean_price:.2f}")
print(f"   ✓ Accrued interest: ${accrued:.2f}")
print(f"   ✓ Price as % of par: {clean_price/1000*100:.3f}%")

# Test 4: Calculate risk metrics
print("\n4. Calculating risk metrics...")
mod_duration = engine.modified_duration(bond)
convexity = engine.convexity(bond)
dv01 = engine.dv01(bond)

print(f"   ✓ Modified duration: {mod_duration:.4f} years")
print(f"   ✓ Convexity: {convexity:.4f}")
print(f"   ✓ DV01: ${dv01:.4f}")

# Test 5: Calculate YTM
print("\n5. Calculating yield to maturity...")
ytm = engine.yield_to_maturity(bond, clean_price, clean_price=True)
print(f"   ✓ Yield to maturity: {ytm:.4%}")

# Test 6: Test par bond (coupon = yield)
print("\n6. Testing par bond pricing (coupon = yield)...")
rate_curve_par = FlatRateCurve(rate=0.05)
pricing_env_par = PricingEnvironment(
    rate_curve=rate_curve_par,
    valuation_date=valuation_date
)
engine_par = BondDiscountEngine(pricing_env_par)
par_price = engine_par.clean_price(bond, valuation_date, valuation_date)
print(f"   ✓ Par bond price: ${par_price:.2f}")
print(f"   ✓ Deviation from par: ${abs(par_price - 1000.0):.2f}")
assert abs(par_price - 1000.0) < 10.0, "Par bond should price near par"

# Test 7: Test interpolated curves
print("\n7. Testing interpolated rate curves...")
pillars = [
    (0.5, 0.030),
    (1.0, 0.035),
    (2.0, 0.040),
    (5.0, 0.045),
    (10.0, 0.050),
]
linear_curve = LinearRateCurve(pillars)
pricing_env_linear = PricingEnvironment(
    rate_curve=linear_curve,
    valuation_date=valuation_date
)
engine_linear = BondDiscountEngine(pricing_env_linear)
linear_price = engine_linear.clean_price(bond)
print(f"   ✓ Price with linear curve: ${linear_price:.2f}")

# Test 8: Test different payment frequencies
print("\n8. Testing different payment frequencies...")
frequencies = [
    PaymentFrequency.ANNUAL,
    PaymentFrequency.SEMI_ANNUAL,
    PaymentFrequency.QUARTERLY,
]

for freq in frequencies:
    bond_freq = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.06,
        payment_frequency=freq
    )
    price_freq = engine.clean_price(bond_freq)
    print(f"   ✓ {freq.name}: ${price_freq:.2f}")

# Test 9: Accrued interest calculation
print("\n9. Testing accrued interest calculation...")
settlement = datetime(2024, 4, 1)
accrued_mid = bond.calculate_accrued_interest(settlement)
print(f"   ✓ Accrued interest at {settlement.date()}: ${accrued_mid:.2f}")
assert 10.0 < accrued_mid < 30.0, "Accrued should be reasonable"

# Test 10: Premium vs discount
print("\n10. Testing premium/discount bonds...")
# Premium bond (coupon > yield)
rate_low = FlatRateCurve(rate=0.03)
env_low = PricingEnvironment(rate_curve=rate_low, valuation_date=valuation_date)
engine_low = BondDiscountEngine(env_low)
premium_price = engine_low.clean_price(bond, valuation_date, valuation_date)
print(f"   ✓ Premium bond (5% coupon, 3% yield): ${premium_price:.2f}")
assert premium_price > 1000.0, "Premium bond should be above par"

# Discount bond (coupon < yield)
rate_high = FlatRateCurve(rate=0.07)
env_high = PricingEnvironment(rate_curve=rate_high, valuation_date=valuation_date)
engine_high = BondDiscountEngine(env_high)
discount_price = engine_high.clean_price(bond, valuation_date, valuation_date)
print(f"   ✓ Discount bond (5% coupon, 7% yield): ${discount_price:.2f}")
assert discount_price < 1000.0, "Discount bond should be below par"

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 80)

