# Plotting Style Guide

Comprehensive visual standards for risk metric analysis visualizations in QuantArk.

---

## Table of Contents

1. [Color Strategy](#color-strategy)
2. [Typography](#typography)
3. [Layout & Grid](#layout--grid)
4. [Figure Sizing](#figure-sizing)
5. [Line Styles & Markers](#line-styles--markers)
6. [Annotations & Labels](#annotations--labels)
7. [Colorbars & Legends](#colorbars--legends)
8. [3D Plot Standards](#3d-plot-standards)
9. [Accessibility](#accessibility)
10. [Code Templates](#code-templates)

---

## Color Strategy

### Sequential Colormaps (Single Direction)

Use for values that go from low to high without a meaningful center point.

| Colormap | Use Case | Example Greeks |
|----------|----------|----------------|
| `viridis` | **Default** - General purpose, perceptually uniform | Price, Vega, Gamma |
| `plasma` | Higher contrast needed | Large value ranges |
| `cividis` | Colorblind-safe alternative | Accessibility priority |
| `YlOrRd` | Heat/intensity emphasis | Risk concentrations |

```python
# Sequential colormap usage
plt.pcolormesh(X, Y, Z, cmap='viridis')
```

### Diverging Colormaps (Centered at Zero)

Use when zero is a meaningful reference point and positive/negative values have different implications.

| Colormap | Use Case | Example Greeks |
|----------|----------|----------------|
| `RdBu_r` | **Default diverging** - Red=negative, Blue=positive | Delta, Theta |
| `coolwarm` | Softer contrast | Presentation-friendly |
| `PiYG` | Alternative diverging | When red/blue is confusing |
| `seismic` | Maximum contrast at extremes | Highlighting outliers |

```python
from matplotlib.colors import TwoSlopeNorm

# Diverging colormap centered at zero
vmin, vmax = data.min(), data.max()
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
plt.pcolormesh(X, Y, Z, cmap='RdBu_r', norm=norm)
```

### Greek-Specific Color Assignments

| Greek | Colormap | Rationale |
|-------|----------|-----------|
| **Price** | `viridis` | Always positive, sequential |
| **Delta** | `RdBu_r` | Sign matters (long/short exposure) |
| **Gamma** | `viridis` | Usually positive, peaks ATM |
| **Vega** | `viridis` | Usually positive for long options |
| **Theta** | `viridis_r` or `RdBu_r` | Usually negative (decay), reverse for intuition |
| **Rho** | `RdBu_r` | Sign differs for calls/puts |
| **DV01** | `RdBu_r` | Direction matters for hedging |

### Categorical Colors (Multiple Series)

Use for comparing multiple products or scenarios on the same plot.

```python
# QuantArk standard palette (colorblind-friendly)
QUANTARK_COLORS = {
    'primary': '#1f77b4',    # Blue - base case
    'secondary': '#ff7f0e',  # Orange - comparison
    'tertiary': '#2ca02c',   # Green - alternative
    'quaternary': '#d62728', # Red - warning/risk
    'highlight': '#9467bd',  # Purple - special case
}

# For multiple lines
SERIES_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
```

### Marker Colors

| Element | Color | Hex |
|---------|-------|-----|
| Current position | Red | `#d62728` |
| Strike line | White (on heatmap) / Red (on line plot) | `#ffffff` / `#d62728` |
| ATM reference | Gray | `#7f7f7f` |
| Barrier level | Orange | `#ff7f0e` |
| Warning zone | Light red | `#ffcccc` |

---

## Typography

### Font Family

```python
# Set global font family
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

# For mathematical expressions
plt.rcParams['mathtext.fontset'] = 'dejavusans'
```

### Font Sizes

| Element | Size | Weight | Example |
|---------|------|--------|---------|
| Figure title | 14pt | Bold | `fontsize=14, fontweight='bold'` |
| Subplot title | 12pt | Bold | `fontsize=12, fontweight='bold'` |
| Axis labels | 11-12pt | Normal | `fontsize=11` |
| Tick labels | 10pt | Normal | `plt.tick_params(labelsize=10)` |
| Legend | 10pt | Normal | `fontsize=10` |
| Annotations | 9-10pt | Normal | `fontsize=9` |
| Colorbar label | 11pt | Normal | `fontsize=11` |

```python
# Apply consistent typography
def apply_typography(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
```

### Greek Symbols

Always use proper Greek letters in labels:

```python
# Correct Greek symbol usage
GREEK_LABELS = {
    'delta': r'Delta ($\Delta$)',
    'gamma': r'Gamma ($\Gamma$)',
    'vega': r'Vega ($\nu$)',
    'theta': r'Theta ($\Theta$)',
    'rho': r'Rho ($\rho$)',
}

# In axis labels
ax.set_ylabel(r'$\Delta$ (Delta)', fontsize=11)
ax.set_ylabel(r'$\Gamma$ (Gamma)', fontsize=11)
```

---

## Layout & Grid

### Grid Spacing

```python
# Standard grid appearance
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# For financial plots (subtle grid)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

# Major and minor grid
ax.grid(True, which='major', alpha=0.3, linestyle='-')
ax.grid(True, which='minor', alpha=0.1, linestyle=':')
ax.minorticks_on()
```

### Multi-Panel Layout

```python
from matplotlib.gridspec import GridSpec

# Standard 2x2 layout
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

# Asymmetric layout (main + sidebar)
gs = GridSpec(2, 3, figure=fig, 
              width_ratios=[2, 2, 1],
              height_ratios=[1, 1],
              hspace=0.25, wspace=0.20)
```

### Margin Standards

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `hspace` | 0.25-0.35 | Vertical spacing between subplots |
| `wspace` | 0.20-0.30 | Horizontal spacing between subplots |
| `left` | 0.08-0.10 | Left margin |
| `right` | 0.92-0.95 | Right margin |
| `top` | 0.92-0.95 | Top margin |
| `bottom` | 0.08-0.10 | Bottom margin |

```python
# Fine-tune layout
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08,
                    hspace=0.30, wspace=0.25)

# Or use tight_layout with padding
plt.tight_layout(pad=1.5)
```

---

## Figure Sizing

### Standard Sizes

| Plot Type | Size (inches) | Aspect Ratio |
|-----------|---------------|--------------|
| Single panel | (10, 7) | ~1.43:1 |
| Single panel (wide) | (12, 6) | 2:1 |
| 2x2 grid | (14, 12) | ~1.17:1 |
| 1x3 horizontal | (16, 5) | 3.2:1 |
| 3x1 vertical | (8, 14) | ~0.57:1 |
| Heatmap (square data) | (10, 9) | ~1.11:1 |
| 3D surface | (12, 10) | 1.2:1 |
| Report figure | (10, 8) | 1.25:1 |

```python
# Figure size by plot type
FIGURE_SIZES = {
    'single': (10, 7),
    'heatmap': (12, 10),
    '3d_surface': (14, 10),
    'multi_2x2': (14, 12),
    'multi_1x3': (16, 5),
    'report': (10, 8),
}

fig, ax = plt.subplots(figsize=FIGURE_SIZES['heatmap'])
```

### DPI Settings

| Output | DPI | Purpose |
|--------|-----|---------|
| Screen/notebook | 100 | Quick preview |
| Report (standard) | 150 | Good balance |
| Publication | 300 | High quality print |
| Presentation | 150 | Slide decks |

```python
# Save with appropriate DPI
plt.savefig('output.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
```

---

## Line Styles & Markers

### Line Styles

| Element | Style | Width | Alpha |
|---------|-------|-------|-------|
| Primary data | Solid `-` | 2.0 | 1.0 |
| Secondary data | Dashed `--` | 1.5 | 0.9 |
| Reference line | Dotted `:` | 1.5 | 0.7 |
| Strike/barrier | Dash-dot `-.` | 1.5 | 0.8 |
| Grid | Solid `-` | 0.5 | 0.3 |
| Contour | Solid `-` | 0.5 | 0.5 |

```python
# Line style constants
LINE_STYLES = {
    'primary': {'linestyle': '-', 'linewidth': 2.0, 'alpha': 1.0},
    'secondary': {'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.9},
    'reference': {'linestyle': ':', 'linewidth': 1.5, 'alpha': 0.7},
    'strike': {'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.8},
}

ax.plot(x, y, color='#1f77b4', **LINE_STYLES['primary'], label='Delta')
ax.axvline(x=strike, color='red', **LINE_STYLES['strike'], label='Strike')
```

### Markers

| Element | Marker | Size | Edge |
|---------|--------|------|------|
| Current position | `★` (star) | 150-200 | White, width=2 |
| Data points | `o` (circle) | 50-80 | None or black |
| Highlighted point | `s` (square) | 100 | White, width=1.5 |
| Observation dates | `|` (vline) | 80 | None |

```python
# Marker constants
MARKERS = {
    'current': {'marker': '★', 's': 200, 'edgecolors': 'white', 'linewidths': 2, 'zorder': 10},
    'data_point': {'marker': 'o', 's': 50, 'edgecolors': 'black', 'linewidths': 0.5},
    'highlight': {'marker': 's', 's': 100, 'edgecolors': 'white', 'linewidths': 1.5},
}

ax.scatter([x], [y], c='red', **MARKERS['current'], label='Current')
```

---

## Annotations & Labels

### Title Format

```python
# Main title format
title = f"{greek_name.capitalize()} Analysis: {product_name}\n(K={strike}, T={maturity}yr, σ={vol:.0%})"
ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

# Multi-line with parameters
title_lines = [
    f"{greek_name.capitalize()} Heatmap: Spot × Volatility",
    f"Strike={strike}, Maturity={maturity}yr"
]
ax.set_title('\n'.join(title_lines), fontsize=12, fontweight='bold')
```

### Axis Label Format

```python
# Standard formats
ax.set_xlabel('Spot Price ($)', fontsize=11)
ax.set_ylabel('Volatility (%)', fontsize=11)
ax.set_ylabel(r'Delta ($\Delta$)', fontsize=11)

# With units
ax.set_ylabel('Theta ($ per day)', fontsize=11)
ax.set_ylabel('Vega ($ per 1% vol)', fontsize=11)
ax.set_ylabel('DV01 ($ per bp)', fontsize=11)
```

### Number Formatting

```python
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Currency format
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Percentage format
ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # For 0-1 values
ax.yaxis.set_major_formatter(PercentFormatter(100))  # For 0-100 values

# Scientific notation for small numbers
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2e}' if abs(x) < 0.001 else f'{x:.4f}'))
```

### Text Annotations

```python
# Annotation style
ax.annotate(
    f'Max: {max_val:.4f}',
    xy=(max_x, max_y),
    xytext=(10, 10),
    textcoords='offset points',
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
)

# Simple text box
ax.text(
    0.02, 0.98, f'Current: {value:.4f}',
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)
```

---

## Colorbars & Legends

### Colorbar Positioning

```python
# Standard colorbar
cbar = plt.colorbar(im, ax=ax, label=r'$\Delta$', shrink=0.9, aspect=30)
cbar.ax.tick_params(labelsize=9)

# Horizontal colorbar (below plot)
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                    label='Delta', shrink=0.8, pad=0.12)

# For 3D plots
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
```

### Legend Placement

| Plot Type | Location | Parameters |
|-----------|----------|------------|
| Line plot | Upper right | `loc='upper right'` |
| Multiple series | Outside right | `loc='center left', bbox_to_anchor=(1.02, 0.5)` |
| Heatmap | Upper right | `loc='upper right'` |
| 3D plot | Upper left | `loc='upper left'` |

```python
# Standard legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Legend outside plot
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
          fontsize=10, framealpha=0.9)
fig.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend

# Multiple columns
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          ncol=3, fontsize=9)
```

---

## 3D Plot Standards

### View Angle

```python
# Recommended view angles by surface shape
VIEW_ANGLES = {
    'default': (elev=25, azim=45),
    'front': (elev=0, azim=0),
    'top': (elev=90, azim=0),
    'isometric': (elev=35, azim=45),
    'presentation': (elev=20, azim=-60),
}

ax.view_init(**VIEW_ANGLES['default'])
```

### Surface Appearance

```python
# Standard surface settings
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',
    edgecolor='none',      # No wireframe
    alpha=0.9,             # Slight transparency
    rstride=1,             # Row stride (resolution)
    cstride=1,             # Column stride
    linewidth=0,
    antialiased=True
)

# With wireframe overlay
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.3, alpha=0.3)
```

### Axis Settings for 3D

```python
# Consistent 3D appearance
ax.set_xlabel('Spot', fontsize=11, labelpad=10)
ax.set_ylabel('Vol (%)', fontsize=11, labelpad=10)
ax.set_zlabel('Gamma', fontsize=11, labelpad=10)

# Remove panes for cleaner look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Lighter grid
ax.xaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.5)
ax.yaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.5)
ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.5)
```

---

## Accessibility

### Colorblind-Safe Palettes

```python
# Colorblind-friendly categorical palette
COLORBLIND_PALETTE = [
    '#0077BB',  # Blue
    '#EE7733',  # Orange
    '#009988',  # Teal
    '#CC3311',  # Red
    '#33BBEE',  # Cyan
    '#EE3377',  # Magenta
    '#BBBBBB',  # Gray
]

# Colorblind-safe colormaps
SAFE_SEQUENTIAL = 'cividis'  # Perceptually uniform, colorblind-safe
SAFE_DIVERGING = 'PuOr'      # Alternative to RdBu
```

### Contrast Requirements

```python
# Ensure sufficient contrast
# - Text on white: use colors darker than #767676
# - White text: use backgrounds darker than #767676

# Good contrast pairs
CONTRAST_PAIRS = {
    'dark_on_light': ('#333333', '#ffffff'),  # Dark gray on white
    'light_on_dark': ('#ffffff', '#1f1f1f'),  # White on dark
    'accent': ('#d62728', '#ffffff'),          # Red on white
}
```

### Alternative Indicators

Always use shape/pattern in addition to color:

```python
# Use different markers for different series
markers = ['o', 's', '^', 'D', 'v']  # circle, square, triangle, diamond
linestyles = ['-', '--', ':', '-.']

for i, (data, label) in enumerate(series):
    ax.plot(x, data, marker=markers[i % 5], linestyle=linestyles[i % 4],
            color=COLORBLIND_PALETTE[i], label=label)
```

---

## Code Templates

### Complete Style Configuration

```python
"""
QuantArk Risk Metric Plotting Style Configuration
Apply at the beginning of any plotting script.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_quantark_style():
    """Apply QuantArk standard plotting style."""
    
    # Figure
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.edgecolor'] = 'white'
    
    # Font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.size'] = 10
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    
    # Axes
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True
    
    # Grid
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.linestyle'] = '-'
    
    # Ticks
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    
    # Legend
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '0.8'
    
    # Lines
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 6
    
    # Saving
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'none'

# Apply on import
apply_quantark_style()
```

### Heatmap Template Function

```python
def create_greek_heatmap(x_values, y_values, z_values, 
                         xlabel, ylabel, greek_name,
                         title=None, mark_current=None, mark_strike=None,
                         output_path=None):
    """
    Create a standardized Greek heatmap.
    
    Parameters
    ----------
    x_values : array-like
        X-axis values (typically spot prices)
    y_values : array-like  
        Y-axis values (typically volatility or time)
    z_values : 2D array
        Greek values on the grid
    xlabel, ylabel : str
        Axis labels
    greek_name : str
        Name of the Greek for colormap selection
    title : str, optional
        Plot title
    mark_current : tuple, optional
        (x, y) coordinates to mark current position
    mark_strike : float, optional
        Strike price to mark with vertical line
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    from matplotlib.colors import TwoSlopeNorm
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select colormap
    vmin, vmax = np.nanmin(z_values), np.nanmax(z_values)
    if greek_name in ['delta', 'theta', 'rho'] and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = 'RdBu_r'
    elif greek_name == 'theta':
        norm = None
        cmap = 'viridis_r'
    else:
        norm = None
        cmap = 'viridis'
    
    # Create heatmap
    X, Y = np.meshgrid(x_values, y_values)
    im = ax.pcolormesh(x_values, y_values, z_values, 
                       cmap=cmap, norm=norm, shading='auto')
    
    # Add contours
    try:
        contour = ax.contour(X, Y, z_values, levels=10, 
                            colors='black', linewidths=0.5, alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
    except:
        pass
    
    # Mark current position
    if mark_current:
        ax.scatter([mark_current[0]], [mark_current[1]], 
                   c='red', s=200, marker='★',
                   edgecolors='white', linewidths=2,
                   zorder=5, label='Current')
    
    # Mark strike
    if mark_strike:
        ax.axvline(x=mark_strike, color='white', linestyle='--',
                   linewidth=1.5, alpha=0.8, label='Strike')
    
    # Colorbar and labels
    cbar = plt.colorbar(im, ax=ax, label=greek_name.capitalize())
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    if mark_current or mark_strike:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax
```

### 3D Surface Template Function

```python
def create_greek_surface(x_values, y_values, z_values,
                         xlabel, ylabel, zlabel,
                         title=None, mark_current=None,
                         view_angle=(25, 45), output_path=None):
    """
    Create a standardized 3D Greek surface plot.
    
    Parameters
    ----------
    x_values, y_values : array-like
        Grid coordinates
    z_values : 2D array
        Surface heights (Greek values)
    xlabel, ylabel, zlabel : str
        Axis labels
    title : str, optional
        Plot title
    mark_current : tuple, optional
        (x, y, z) coordinates to mark current position
    view_angle : tuple
        (elevation, azimuth) for viewing angle
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig, ax : matplotlib figure and 3D axis
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(x_values, y_values)
    
    # Surface plot
    surf = ax.plot_surface(
        X, Y, z_values,
        cmap=cm.viridis,
        edgecolor='none',
        alpha=0.9,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True
    )
    
    # Mark current position
    if mark_current:
        ax.scatter([mark_current[0]], [mark_current[1]], [mark_current[2]],
                   c='red', s=200, marker='★',
                   edgecolors='white', linewidths=2,
                   zorder=10, label='Current')
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, 
                        label=zlabel, pad=0.1)
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=11, labelpad=10)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    if mark_current:
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax
```

---

## Quick Reference Card

| Element | Setting |
|---------|---------|
| **Figure size** | Single: (10,7), Heatmap: (12,10), 3D: (14,10), Grid 2×2: (14,12) |
| **DPI** | Screen: 100, Report: 150, Publication: 300 |
| **Title font** | 14pt bold |
| **Axis label** | 11pt normal |
| **Tick labels** | 10pt normal |
| **Line width** | Primary: 2.0, Secondary: 1.5, Reference: 1.5 |
| **Grid alpha** | 0.3 |
| **Sequential cmap** | `viridis` (default) |
| **Diverging cmap** | `RdBu_r` with `TwoSlopeNorm` |
| **Current marker** | Red star (★), size=200, white edge |
| **Strike line** | Dashed, white on heatmap, red on line plot |
| **3D view angle** | elev=25, azim=45 |
