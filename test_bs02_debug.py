from asset.equity.product.option import AmericanOption
from asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from datetime import datetime
import numpy as np
from scipy.stats import norm

# Manually test BS02 calculation
S, K, T = 100.0, 100.0, 1.0
r, q, sigma = 0.05, 0.02, 0.20
b = r - q

print('Manual BS02 calculation:')
print('S=%s, K=%s, T=%s, r=%s, q=%s, b=%s, sigma=%s' % (S, K, T, r, q, b, sigma))
print()

# Calculate beta
beta = (0.5 - b / sigma**2) + np.sqrt((0.5 - b / sigma**2)**2 + 2 * r / sigma**2)
print('beta = %.6f' % beta)

# Calculate boundaries
B_infinity = beta * K / (beta - 1.0)
B_0 = max(K, r * K / (r - b))
print('B_infinity = %.6f' % B_infinity)
print('B_0 = %.6f' % B_0)

# Calculate t1
t1 = 0.5 * (np.sqrt(5) - 1) * T
print('t1 = %.6f' % t1)

# Calculate h1 and h2
h1 = -(b * t1 + 2 * sigma * np.sqrt(t1)) * K**2 / ((B_infinity - B_0) * B_0)
h2 = -(b * T + 2 * sigma * np.sqrt(T)) * K**2 / ((B_infinity - B_0) * B_0)
print('h1 = %.6f' % h1)
print('h2 = %.6f' % h2)

# Calculate I1 and I2
I1 = B_0 + (B_infinity - B_0) * (1 - np.exp(h1))
I2 = B_0 + (B_infinity - B_0) * (1 - np.exp(h2))
print('I1 = %.6f' % I1)
print('I2 = %.6f' % I2)

# Check early exercise condition
print()
print('S >= I2? %s' % (S >= I2))
print('If true, immediate exercise value = S - K = %.6f' % (S - K))
