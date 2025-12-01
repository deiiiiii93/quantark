from asset.equity.product.option import AmericanOption
from asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from datetime import datetime
import numpy as np
from scipy.stats import norm

# Test the specific case causing warnings
pricing_env = PricingEnvironment(
    spot_quote=SpotQuote(spot=100.0),
    vol_surface=FlatVolSurface(volatility=0.20),
    rate_curve=FlatRateCurve(rate=0.05),
    div_yield=ContinuousDividendYield(div_yield=0.02),
    valuation_date=datetime(2024, 1, 1),
)

call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)

print('Case causing BS02 warning:')
print('S=100, K=100, r=5%, q=2%, sigma=20%, T=1yr')
print()

# Manually calculate what BS02 should produce
S, K, T = 100.0, 100.0, 1.0
r, q, sigma = 0.05, 0.02, 0.20
b = r - q

beta = (0.5 - b / sigma**2) + np.sqrt((0.5 - b / sigma**2)**2 + 2 * r / sigma**2)
B_infinity = beta * K / (beta - 1.0)
B_0 = max(K, r * K / (r - b))
t1 = 0.5 * (np.sqrt(5) - 1) * T
h1 = -(b * t1 + 2 * sigma * np.sqrt(t1)) * K**2 / ((B_infinity - B_0) * B_0)
h2 = -(b * T + 2 * sigma * np.sqrt(T)) * K**2 / ((B_infinity - B_0) * B_0)
I1 = B_0 + (B_infinity - B_0) * (1 - np.exp(h1))
I2 = B_0 + (B_infinity - B_0) * (1 - np.exp(h2))

print('Calculated parameters:')
print('beta = %.6f' % beta)
print('B_0 = %.6f, B_inf = %.6f' % (B_0, B_infinity))
print('I1 = %.6f, I2 = %.6f' % (I1, I2))
print()

if I1 > K:
    log_alpha1 = np.log(I1 - K) - beta * np.log(I1)
    alpha1 = np.exp(log_alpha1) if log_alpha1 > -700 else 0.0
else:
    alpha1 = 0.0

if I2 > K:
    log_alpha2 = np.log(I2 - K) - beta * np.log(I2)
    alpha2 = np.exp(log_alpha2) if log_alpha2 > -700 else 0.0
else:
    alpha2 = 0.0

print('alpha1 = %.6e (log_alpha1 = %.6f)' % (alpha1, log_alpha1 if I1 > K else float('nan')))
print('alpha2 = %.6e (log_alpha2 = %.6f)' % (alpha2, log_alpha2 if I2 > K else float('nan')))
print()

# The first term alone
print('First term: alpha2 * S^beta = %.6e * %.6e = %.6f' % (alpha2, S**beta, alpha2 * S**beta))
print()

# Now test with engine
engine = AmericanOptionAnalyticalEngine(method='BS02')
price = engine.price(call, pricing_env)
print('BS02 actual price: $%.6f' % price)

# Compare to BS93
engine_bs93 = AmericanOptionAnalyticalEngine(method='BS93')
price_bs93 = engine_bs93.price(call, pricing_env)
print('BS93 price: $%.6f' % price_bs93)
