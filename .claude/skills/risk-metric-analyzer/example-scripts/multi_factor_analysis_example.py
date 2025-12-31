"""
Risk Metric Analysis Example: Multi-Factor Visualization
=========================================================

This script demonstrates multi-factor risk analysis using 2D heatmaps
and 3D surface plots to visualize how Greeks change across multiple
market parameters simultaneously.

Analysis includes:
1. Delta heatmap: Spot × Volatility
2. Gamma 3D surface: Spot × Volatility
3. Theta heatmap: Spot × Time-to-Maturity
4. Multi-factor CSV data export

Usage:
    python risk_metric_analysis/risk_metric_calculation_scripts/multi_factor_analysis_example.py
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime
import pandas as pd
import os
from copy import deepcopy

# QuantArk imports
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.riskmeasures import GreeksCalculator
from asset.equity.param import EngineParams
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType


# === QuantArk Plotting Style ===
def apply_quantark_style():
    """Apply QuantArk standard plotting style (from plotting-style-guide.md)."""
    
    # Figure defaults
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.edgecolor'] = 'white'
    
    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.size'] = 10
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    
    # Axes settings
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True
    
    # Grid settings
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.linestyle'] = '-'
    
    # Tick settings
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Legend settings
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.framealpha'] = 0.9
    
    # Line settings
    plt.rcParams['lines.linewidth'] = 2.0
    
    # Save settings
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.facecolor'] = 'white'


# Apply style on import
apply_quantark_style()


# === Style Constants ===
# Colors (from plotting-style-guide.md)
QUANTARK_COLORS = {
    'primary': '#1f77b4',    # Blue - base case
    'secondary': '#ff7f0e',  # Orange - comparison
    'current_marker': '#d62728',  # Red - current position
    'strike_line': '#ffffff',     # White on heatmaps
}

# Marker settings
CURRENT_MARKER = {
    'marker': '★',
    's': 200,
    'edgecolors': 'white',
    'linewidths': 2,
    'zorder': 10
}


# === Configuration ===
# Product Parameters
STRIKE = 100.0
MATURITY = 1.0  # 1 year
OPTION_TYPE = OptionType.CALL

# Market Data (Base Case)
SPOT = 100.0       # ATM option
VOLATILITY = 0.20  # 20% implied vol
RATE = 0.05        # 5% risk-free rate
DIV_YIELD = 0.02   # 2% dividend yield

# Grid Parameters
SPOT_RANGE = (70, 130)    # Spot range
VOL_RANGE = (0.10, 0.50)  # 10% to 50% volatility
TIME_RANGE = (0.02, 1.0)  # 1 week to 1 year
GRID_SIZE = 25            # Grid resolution

# Output directory
OUTPUT_DIR = 'risk_metric_analysis/reports/'


def create_pricing_env(spot, vol, rate=RATE, div_yield=DIV_YIELD):
    """Helper to create pricing environment with given parameters."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime.now(),
    )


def calculate_2d_grid(product, engine, calculator, x_values, y_type, y_values, greek_name):
    """
    Calculate Greeks on a 2D grid.
    
    Args:
        product: Base product (will be modified for time dimension)
        engine: Pricing engine
        calculator: Greeks calculator
        x_values: Spot values (X-axis)
        y_type: 'vol' or 'time'
        y_values: Volatility or time values (Y-axis)
        greek_name: Name of Greek to calculate
        
    Returns:
        2D numpy array of Greek values
    """
    greek_grid = np.zeros((len(y_values), len(x_values)))
    
    for i, y_val in enumerate(y_values):
        if y_type == 'vol':
            # Varying volatility
            current_product = product
            for j, spot in enumerate(x_values):
                env = create_pricing_env(spot, y_val)
                try:
                    greeks = calculator.calculate_numerical_greeks(current_product, env, engine)
                    greek_grid[i, j] = greeks[greek_name]
                except Exception:
                    greek_grid[i, j] = np.nan
        else:
            # Varying time - need to create new product with different maturity
            current_product = EuropeanVanillaOption(
                strike=STRIKE,
                option_type=OPTION_TYPE,
                maturity=y_val
            )
            for j, spot in enumerate(x_values):
                env = create_pricing_env(spot, VOLATILITY)
                try:
                    greeks = calculator.calculate_numerical_greeks(current_product, env, engine)
                    greek_grid[i, j] = greeks[greek_name]
                except Exception:
                    greek_grid[i, j] = np.nan
    
    return greek_grid


def plot_heatmap(x_values, y_values, z_values, xlabel, ylabel, title, 
                 greek_name, output_path, mark_current=None):
    """Create a 2D heatmap visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Choose colormap based on Greek
    vmin, vmax = np.nanmin(z_values), np.nanmax(z_values)
    if greek_name in ['delta', 'theta'] and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = 'RdBu_r'
    elif greek_name == 'theta':
        cmap = 'viridis_r'  # Reverse for theta (usually negative)
        norm = None
    else:
        cmap = 'viridis'
        norm = None
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_values, y_values)
    
    im = ax.pcolormesh(x_values, y_values, z_values, cmap=cmap, norm=norm, shading='auto')
    
    # Add contour lines
    try:
        contour = ax.contour(X, Y, z_values, levels=10, colors='black', 
                             linewidths=0.5, alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
    except Exception:
        pass  # Skip contours if they fail
    
    # Mark current position
    if mark_current:
        ax.scatter([mark_current[0]], [mark_current[1]], c='red', s=200, 
                   marker='★', edgecolors='white', linewidths=2, 
                   zorder=5, label='Current')
    
    # Mark strike
    ax.axvline(x=STRIKE, color='white', linestyle='--', linewidth=1.5, 
               alpha=0.8, label='Strike')
    
    # Colorbar and labels
    cbar = plt.colorbar(im, ax=ax, label=f'{greek_name.capitalize()}')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_3d_surface(x_values, y_values, z_values, xlabel, ylabel, zlabel,
                    title, output_path, mark_current=None):
    """Create a 3D surface visualization."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(x_values, y_values)
    
    # Surface plot
    surf = ax.plot_surface(
        X, Y, z_values,
        cmap=cm.viridis, edgecolor='none', alpha=0.9,
        rstride=1, cstride=1, linewidth=0, antialiased=True
    )
    
    # Mark current position
    if mark_current:
        ax.scatter([mark_current[0]], [mark_current[1]], [mark_current[2]],
                   c='red', s=200, marker='★', edgecolors='white',
                   linewidths=2, zorder=10, label='Current')
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.view_init(elev=25, azim=45)
    if mark_current:
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def export_grid_to_csv(x_values, y_values, z_values, x_name, y_name, z_name, output_path):
    """Export 2D grid data to CSV."""
    rows = []
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            rows.append({
                x_name: x_val,
                y_name: y_val,
                z_name: z_values[i, j]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("MULTI-FACTOR RISK ANALYSIS")
    print("European Vanilla Call Option")
    print("=" * 70 + "\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup
    product = EuropeanVanillaOption(
        strike=STRIKE,
        option_type=OPTION_TYPE,
        maturity=MATURITY
    )
    engine = BlackScholesEngine()
    calculator = GreeksCalculator(params=EngineParams(bump_size=0.01))
    
    # Define grids
    spot_grid = np.linspace(*SPOT_RANGE, GRID_SIZE)
    vol_grid = np.linspace(*VOL_RANGE, GRID_SIZE)
    time_grid = np.linspace(*TIME_RANGE, GRID_SIZE)
    
    # Calculate current Greeks for reference
    base_env = create_pricing_env(SPOT, VOLATILITY)
    base_greeks = calculator.calculate_analytical_greeks(product, base_env)
    
    print("BASE CASE GREEKS")
    print("-" * 40)
    print(f"  Price:  {base_greeks['price']:.6f}")
    print(f"  Delta:  {base_greeks['delta']:.6f}")
    print(f"  Gamma:  {base_greeks['gamma']:.6f}")
    print(f"  Vega:   {base_greeks['vega']:.6f}")
    print(f"  Theta:  {base_greeks['theta']:.6f}")
    print()
    
    # ================================================================
    # 1. Delta Heatmap: Spot × Volatility
    # ================================================================
    print("Calculating Delta grid (Spot × Volatility)...")
    delta_grid = calculate_2d_grid(
        product, engine, calculator,
        spot_grid, 'vol', vol_grid, 'delta'
    )
    
    plot_heatmap(
        spot_grid, vol_grid * 100, delta_grid,
        xlabel='Spot Price', ylabel='Volatility (%)',
        title=f'Delta Heatmap: Spot × Volatility\n(Strike={STRIKE}, T={MATURITY}yr)',
        greek_name='delta',
        output_path=os.path.join(OUTPUT_DIR, 'delta_spot_vol_heatmap.png'),
        mark_current=(SPOT, VOLATILITY * 100)
    )
    
    export_grid_to_csv(
        spot_grid, vol_grid, delta_grid,
        'Spot', 'Volatility', 'Delta',
        os.path.join(OUTPUT_DIR, 'delta_spot_vol_data.csv')
    )
    
    # ================================================================
    # 2. Gamma 3D Surface: Spot × Volatility
    # ================================================================
    print("\nCalculating Gamma grid (Spot × Volatility)...")
    gamma_grid = calculate_2d_grid(
        product, engine, calculator,
        spot_grid, 'vol', vol_grid, 'gamma'
    )
    
    plot_3d_surface(
        spot_grid, vol_grid * 100, gamma_grid,
        xlabel='Spot Price', ylabel='Volatility (%)', zlabel='Gamma',
        title=f'Gamma Surface: Spot × Volatility\n(Strike={STRIKE}, T={MATURITY}yr)',
        output_path=os.path.join(OUTPUT_DIR, 'gamma_spot_vol_surface.png'),
        mark_current=(SPOT, VOLATILITY * 100, base_greeks['gamma'])
    )
    
    # Also create heatmap for gamma
    plot_heatmap(
        spot_grid, vol_grid * 100, gamma_grid,
        xlabel='Spot Price', ylabel='Volatility (%)',
        title=f'Gamma Heatmap: Spot × Volatility\n(Strike={STRIKE}, T={MATURITY}yr)',
        greek_name='gamma',
        output_path=os.path.join(OUTPUT_DIR, 'gamma_spot_vol_heatmap.png'),
        mark_current=(SPOT, VOLATILITY * 100)
    )
    
    export_grid_to_csv(
        spot_grid, vol_grid, gamma_grid,
        'Spot', 'Volatility', 'Gamma',
        os.path.join(OUTPUT_DIR, 'gamma_spot_vol_data.csv')
    )
    
    # ================================================================
    # 3. Theta Heatmap: Spot × Time
    # ================================================================
    print("\nCalculating Theta grid (Spot × Time)...")
    theta_grid = calculate_2d_grid(
        product, engine, calculator,
        spot_grid, 'time', time_grid, 'theta'
    )
    
    plot_heatmap(
        spot_grid, time_grid * 365, theta_grid,  # Convert to days
        xlabel='Spot Price', ylabel='Days to Maturity',
        title=f'Theta Heatmap: Spot × Time\n(Strike={STRIKE}, Vol={VOLATILITY:.0%})',
        greek_name='theta',
        output_path=os.path.join(OUTPUT_DIR, 'theta_spot_time_heatmap.png'),
        mark_current=(SPOT, MATURITY * 365)
    )
    
    export_grid_to_csv(
        spot_grid, time_grid, theta_grid,
        'Spot', 'TimeToMaturity', 'Theta',
        os.path.join(OUTPUT_DIR, 'theta_spot_time_data.csv')
    )
    
    # ================================================================
    # 4. Vega 3D Surface: Spot × Volatility
    # ================================================================
    print("\nCalculating Vega grid (Spot × Volatility)...")
    vega_grid = calculate_2d_grid(
        product, engine, calculator,
        spot_grid, 'vol', vol_grid, 'vega'
    )
    
    plot_3d_surface(
        spot_grid, vol_grid * 100, vega_grid,
        xlabel='Spot Price', ylabel='Volatility (%)', zlabel='Vega (per 1%)',
        title=f'Vega Surface: Spot × Volatility\n(Strike={STRIKE}, T={MATURITY}yr)',
        output_path=os.path.join(OUTPUT_DIR, 'vega_spot_vol_surface.png'),
        mark_current=(SPOT, VOLATILITY * 100, base_greeks['vega'])
    )
    
    # ================================================================
    # Summary Statistics
    # ================================================================
    print("\n" + "=" * 70)
    print("MULTI-FACTOR ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\nDelta Surface Statistics:")
    print(f"  Range: [{np.nanmin(delta_grid):.4f}, {np.nanmax(delta_grid):.4f}]")
    print(f"  Max Delta at: Spot={spot_grid[np.unravel_index(np.nanargmax(delta_grid), delta_grid.shape)[1]]:.1f}, "
          f"Vol={vol_grid[np.unravel_index(np.nanargmax(delta_grid), delta_grid.shape)[0]]*100:.1f}%")
    
    print("\nGamma Surface Statistics:")
    print(f"  Range: [{np.nanmin(gamma_grid):.6f}, {np.nanmax(gamma_grid):.6f}]")
    max_idx = np.unravel_index(np.nanargmax(gamma_grid), gamma_grid.shape)
    print(f"  Max Gamma at: Spot={spot_grid[max_idx[1]]:.1f}, Vol={vol_grid[max_idx[0]]*100:.1f}%")
    
    print("\nTheta Surface Statistics:")
    print(f"  Range: [{np.nanmin(theta_grid):.6f}, {np.nanmax(theta_grid):.6f}]")
    min_idx = np.unravel_index(np.nanargmin(theta_grid), theta_grid.shape)
    print(f"  Min Theta (max decay) at: Spot={spot_grid[min_idx[1]]:.1f}, "
          f"T={time_grid[min_idx[0]]*365:.0f} days")
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"\nHeatmaps:")
    print(f"  - {OUTPUT_DIR}delta_spot_vol_heatmap.png")
    print(f"  - {OUTPUT_DIR}gamma_spot_vol_heatmap.png")
    print(f"  - {OUTPUT_DIR}theta_spot_time_heatmap.png")
    print(f"\n3D Surfaces:")
    print(f"  - {OUTPUT_DIR}gamma_spot_vol_surface.png")
    print(f"  - {OUTPUT_DIR}vega_spot_vol_surface.png")
    print(f"\nCSV Data:")
    print(f"  - {OUTPUT_DIR}delta_spot_vol_data.csv")
    print(f"  - {OUTPUT_DIR}gamma_spot_vol_data.csv")
    print(f"  - {OUTPUT_DIR}theta_spot_time_data.csv")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
