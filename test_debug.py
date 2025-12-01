from asset.equity.product.option import AmericanOption
from asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from datetime import datetime

# Test case that's failing
pricing_env = PricingEnvironment(
    spot_quote=SpotQuote(spot=100.0),
    vol_surface=FlatVolSurface(volatility=0.20),
    rate_curve=FlatRateCurve(rate=0.05),
    div_yield=ContinuousDividendYield(div_yield=0.02),
    valuation_date=datetime(2024, 1, 1),
)

call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)

print('Testing problematic case (ATM call):')
print('S=100, K=100, r=5%, q=2%, sigma=20%, T=1yr')
print()

S, K, T = 100.0, 100.0, 1.0
r, q, sigma = 0.05, 0.02, 0.20
b = r - q

print('b = r - q = %.4f' % b)
print('b >= r? %s' % (b >= r))
print()

# This should NOT use European pricing since b < r
if b >= r:
    print('Should use European pricing (early exercise not optimal)')
else:
    print('Should use American pricing (early exercise may be optimal)')
    
print()

try:
    engine_bs02 = AmericanOptionAnalyticalEngine(method='BS02')
    price_bs02 = engine_bs02.price(call, pricing_env)
    print('BS02 price: $%.6f' % price_bs02)
except Exception as e:
    print('Error: %s' % e)
    import traceback
    traceback.print_exc()

print()
print('Testing with BS93:')
engine_bs93 = AmericanOptionAnalyticalEngine(method='BS93')
price_bs93 = engine_bs93.price(call, pricing_env)
print('BS93 price: $%.6f' % price_bs93)

print()
print('Testing with BAW:')
engine_baw = AmericanOptionAnalyticalEngine(method='BAW')
price_baw = engine_baw.price(call, pricing_env)
print('BAW price: $%.6f' % price_baw)
