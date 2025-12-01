from asset.equity.product.option import AmericanOption
from asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from datetime import datetime

pricing_env = PricingEnvironment(
    spot_quote=SpotQuote(spot=100.0),
    vol_surface=FlatVolSurface(volatility=0.25),
    rate_curve=FlatRateCurve(rate=0.05),
    div_yield=ContinuousDividendYield(div_yield=0.03),
    valuation_date=datetime(2024, 1, 1),
)

call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)

print('American Call Pricing (S=100, K=100, r=5%, q=3%, sigma=25%, T=1yr):')
price_bs93 = AmericanOptionAnalyticalEngine(method='BS93').price(call, pricing_env)
print('BS93: $%.4f' % price_bs93)

price_bs02 = AmericanOptionAnalyticalEngine(method='BS02').price(call, pricing_env)
print('BS02: $%.4f' % price_bs02)

price_baw = AmericanOptionAnalyticalEngine(method='BAW').price(call, pricing_env)
print('BAW:  $%.4f' % price_baw)

print()
put = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

print('American Put Pricing (S=100, K=100, r=5%, q=3%, sigma=25%, T=1yr):')
price_bs93 = AmericanOptionAnalyticalEngine(method='BS93').price(put, pricing_env)
print('BS93: $%.4f' % price_bs93)

price_bs02 = AmericanOptionAnalyticalEngine(method='BS02').price(put, pricing_env)
print('BS02: $%.4f' % price_bs02)

price_baw = AmericanOptionAnalyticalEngine(method='BAW').price(put, pricing_env)
print('BAW:  $%.4f' % price_baw)
