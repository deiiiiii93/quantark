from asset.equity.product.option import AmericanOption
from asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from datetime import datetime
import numpy as np
from scipy.stats import norm

# Manually compute BS02 step by step to find the issue
S, K, T = 100.0, 100.0, 1.0
r, q, sigma = 0.05, 0.02, 0.20
b = r - q

print('BS02 Manual Calculation Debug')
print('=' * 60)
print('Parameters: S=%s, K=%s, T=%s, r=%s, q=%s, b=%s, sigma=%s' % (S, K, T, r, q, b, sigma))
print()

# Step 1: Calculate beta
beta = (0.5 - b / sigma**2) + np.sqrt((0.5 - b / sigma**2)**2 + 2 * r / sigma**2)
print('beta = %.6f' % beta)

# Step 2: Calculate boundaries
B_infinity = beta * K / (beta - 1.0)
B_0 = max(K, r * K / (r - b))
print('B_0 = %.6f' % B_0)
print('B_infinity = %.6f' % B_infinity)

# Step 3: Calculate t1
t1 = 0.5 * (np.sqrt(5) - 1) * T
print('t1 = %.6f' % t1)

# Step 4: Calculate h1, h2
h1 = -(b * t1 + 2 * sigma * np.sqrt(t1)) * K**2 / ((B_infinity - B_0) * B_0)
h2 = -(b * T + 2 * sigma * np.sqrt(T)) * K**2 / ((B_infinity - B_0) * B_0)
print('h1 = %.6f, h2 = %.6f' % (h1, h2))

# Step 5: Calculate I1, I2
I1 = B_0 + (B_infinity - B_0) * (1 - np.exp(h1))
I2 = B_0 + (B_infinity - B_0) * (1 - np.exp(h2))
print('I1 = %.6f, I2 = %.6f' % (I1, I2))

# Step 6: Calculate alphas
if I1 > K:
    log_alpha1 = np.log(I1 - K) - beta * np.log(I1)
    alpha1 = np.exp(log_alpha1)
    print('alpha1 = %.6e (log = %.6f)' % (alpha1, log_alpha1))
else:
    alpha1 = 0.0
    print('alpha1 = 0 (I1 <= K)')

if I2 > K:
    log_alpha2 = np.log(I2 - K) - beta * np.log(I2)
    alpha2 = np.exp(log_alpha2)
    print('alpha2 = %.6e (log = %.6f)' % (alpha2, log_alpha2))
else:
    alpha2 = 0.0
    print('alpha2 = 0 (I2 <= K)')

print()
print('First term alone: alpha2 * S^beta = %.6f' % (alpha2 * S**beta))
print()
print('Now computing phi and psi terms...')
print()

# Define phi function matching reference
def phi_bs02(S, T, gamma, H, I, r, b, sigma):
    if S <= 0 or T <= 0 or H <= 0 or I <= 0:
        return 0.0
    lambda_val = -r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2
    kappa = 2 * b / sigma**2 + 2 * gamma - 1
    d = (np.log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(I**2 / (S * H)) + (b + (gamma - 0.5) * sigma**2) * T) / (sigma * np.sqrt(T))
    term1 = norm.cdf(-d)
    term2 = (I / S)**kappa * norm.cdf(-d2)
    return np.exp(lambda_val * T) * S**gamma * (term1 - term2)

# Compute all 12 terms
term1 = alpha2 * S**beta
term2 = -alpha2 * phi_bs02(S, t1, beta, I2, I2, r, b, sigma)
term3 = phi_bs02(S, t1, 1, I2, I2, r, b, sigma)
term4 = -phi_bs02(S, t1, 1, I1, I2, r, b, sigma)
term5 = -K * phi_bs02(S, t1, 0, I2, I2, r, b, sigma)
term6 = K * phi_bs02(S, t1, 0, I1, I2, r, b, sigma)

print('Terms 1-6 (phi terms):')
print('  term1 = %.6f' % term1)
print('  term2 = %.6f' % term2)
print('  term3 = %.6f' % term3)
print('  term4 = %.6f' % term4)
print('  term5 = %.6f' % term5)
print('  term6 = %.6f' % term6)
print('  Subtotal (1-6) = %.6f' % (term1 + term2 + term3 + term4 + term5 + term6))
