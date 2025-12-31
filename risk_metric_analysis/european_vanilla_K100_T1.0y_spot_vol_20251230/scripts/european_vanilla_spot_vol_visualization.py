"""
Risk Metric Visualization for European Vanilla Option
Spot x Volatility 2D Heatmaps
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from datetime import datetime
import os

from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.riskmeasures import GreeksCalculator
from util.enum import OptionType
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from copy import deepcopy

# === Product Configuration ===
STRIKE = 100.0
MATURITY = 1.0
SPOT = 100.0
VOLATILITY = 0.20
RATE = 0.05
DIV_YIELD = 0.02

# Grid parameters for visualization
SPOT_RANGE = (SPOT * 0.7, SPOT * 1.3)  # +/- 30%
VOL_RANGE = (0.10, 0.50)                # 10% to 50%
GRID_SIZE = 30                          # 30x30 grid

# === Setup Pricing Environment ===
pricing_env = PricingEnvironment(
    spot_quote=SpotQuote(spot=SPOT),
    vol_surface=FlatVolSurface(volatility=VOLATILITY),
    rate_curve=FlatRateCurve(rate=RATE),
    div_yield=ContinuousDividendYield(div_yield=DIV_YIELD),
    valuation_date=datetime(2025, 12, 30),
)

# === Create Products and Engine ===
call_option = EuropeanVanillaOption(strike=STRIKE, option_type=OptionType.CALL, maturity=MATURITY)
put_option = EuropeanVanillaOption(strike=STRIKE, option_type=OptionType.PUT, maturity=MATURITY)
engine = BlackScholesEngine()
calculator = GreeksCalculator()

# === Create Grid ===
spot_grid = np.linspace(*SPOT_RANGE, GRID_SIZE)
vol_grid = np.linspace(*VOL_RANGE, GRID_SIZE)
SPOT_MESH, VOL_MESH = np.meshgrid(spot_grid, vol_grid)

# Greeks to plot
greeks_to_plot = ['delta', 'gamma', 'vega', 'theta']
greek_labels = {'delta': 'Delta', 'gamma': 'Gamma', 'vega': 'Vega', 'theta': 'Theta'}

# === Calculate Greeks for CALL Option ===
print("Calculating CALL option Greeks over spot-volatility grid...")
call_greek_values = {}
for greek_name in greeks_to_plot:
    call_greek_values[greek_name] = np.zeros_like(SPOT_MESH)
    for i, v in enumerate(vol_grid):
        for j, s in enumerate(spot_grid):
            env = deepcopy(pricing_env)
            env.spot_quote = SpotQuote(spot=s)
            env.vol_surface = FlatVolSurface(volatility=v)
            greeks = calculator.calculate_analytical_greeks(call_option, env, engine)
            call_greek_values[greek_name][i, j] = greeks[greek_name]

# === Calculate Greeks for PUT Option ===
print("Calculating PUT option Greeks over spot-volatility grid...")
put_greek_values = {}
for greek_name in greeks_to_plot:
    put_greek_values[greek_name] = np.zeros_like(SPOT_MESH)
    for i, v in enumerate(vol_grid):
        for j, s in enumerate(spot_grid):
            env = deepcopy(pricing_env)
            env.spot_quote = SpotQuote(spot=s)
            env.vol_surface = FlatVolSurface(volatility=v)
            greeks = calculator.calculate_analytical_greeks(put_option, env, engine)
            put_greek_values[greek_name][i, j] = greeks[greek_name]

# === Create Combined Heatmaps for CALL ===
print("Generating CALL option heatmaps...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('European Call Option: Greeks Surface (Spot × Volatility)\nStrike=100, Maturity=1Y, r=5%, q=2%',
             fontsize=14, fontweight='bold')

for idx, greek_name in enumerate(greeks_to_plot):
    ax = axes[idx // 2, idx % 2]
    values = call_greek_values[greek_name]

    # Choose colormap and normalization based on Greek type and value range
    if greek_name in ['delta', 'gamma']:
        # Diverging colormap centered at zero
        if values.min() >= 0 or values.max() <= 0:
            # All positive or all negative - use sequential
            cmap = 'viridis'
            norm = None
        else:
            # Mixed signs - use diverging
            cmap = 'RdBu_r'
            norm = TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    elif greek_name == 'theta':
        # Theta is always negative (time decay) - use sequential
        cmap = 'plasma_r'  # reversed so darker = more negative
        norm = Normalize(vmin=values.min(), vmax=values.max())
    else:  # vega
        # Vega is always positive - use sequential
        cmap = 'viridis'
        norm = Normalize(vmin=values.min(), vmax=values.max())

    im = ax.pcolormesh(SPOT_MESH, VOL_MESH, values, cmap=cmap, norm=norm, shading='auto')
    cbar = fig.colorbar(im, ax=ax, label=greek_labels[greek_name])

    # Mark current position and strike
    ax.axvline(x=SPOT, color='white', linestyle='--', linewidth=1.5, label='Current Spot')
    ax.axvline(x=STRIKE, color='red', linestyle='-', linewidth=1.5, label='Strike')
    ax.axhline(y=VOLATILITY, color='white', linestyle=':', linewidth=1.5, label='Current Vol')

    ax.set_xlabel('Spot Price', fontsize=11)
    ax.set_ylabel('Volatility', fontsize=11)
    ax.set_title(f'{greek_labels[greek_name]}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()

# Get output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
viz_dir = os.path.join(os.path.dirname(script_dir), 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

call_fig_path = os.path.join(viz_dir, 'european_call_spot_vol_heatmaps.png')
plt.savefig(call_fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {call_fig_path}")

# === Create Combined Heatmaps for PUT ===
print("Generating PUT option heatmaps...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('European Put Option: Greeks Surface (Spot × Volatility)\nStrike=100, Maturity=1Y, r=5%, q=2%',
             fontsize=14, fontweight='bold')

for idx, greek_name in enumerate(greeks_to_plot):
    ax = axes[idx // 2, idx % 2]
    values = put_greek_values[greek_name]

    # Choose colormap and normalization based on Greek type and value range
    if greek_name == 'delta':
        # Delta can be negative for puts - check if values span zero
        if values.min() < 0 and values.max() > 0:
            # Mixed signs - use diverging
            cmap = 'RdBu_r'
            norm = TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
        elif values.max() <= 0:
            # All negative - use sequential (plasma reversed for intuitive interpretation)
            cmap = 'plasma_r'
            norm = Normalize(vmin=values.min(), vmax=values.max())
        else:
            # All positive - use sequential
            cmap = 'viridis'
            norm = Normalize(vmin=values.min(), vmax=values.max())
    elif greek_name == 'gamma':
        # Gamma is always positive - use sequential
        cmap = 'viridis'
        norm = Normalize(vmin=values.min(), vmax=values.max())
    elif greek_name == 'theta':
        # Theta is negative - use sequential
        cmap = 'plasma_r'
        norm = Normalize(vmin=values.min(), vmax=values.max())
    else:  # vega
        # Vega is always positive - use sequential
        cmap = 'viridis'
        norm = Normalize(vmin=values.min(), vmax=values.max())

    im = ax.pcolormesh(SPOT_MESH, VOL_MESH, values, cmap=cmap, norm=norm, shading='auto')
    cbar = fig.colorbar(im, ax=ax, label=greek_labels[greek_name])

    # Mark current position and strike
    ax.axvline(x=SPOT, color='white', linestyle='--', linewidth=1.5, label='Current Spot')
    ax.axvline(x=STRIKE, color='red', linestyle='-', linewidth=1.5, label='Strike')
    ax.axhline(y=VOLATILITY, color='white', linestyle=':', linewidth=1.5, label='Current Vol')

    ax.set_xlabel('Spot Price', fontsize=11)
    ax.set_ylabel('Volatility', fontsize=11)
    ax.set_title(f'{greek_labels[greek_name]}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()

put_fig_path = os.path.join(viz_dir, 'european_put_spot_vol_heatmaps.png')
plt.savefig(put_fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {put_fig_path}")

print("\n" + "=" * 60)
print("Visualization complete!")
print(f"Output directory: {viz_dir}/")
print("=" * 60)
