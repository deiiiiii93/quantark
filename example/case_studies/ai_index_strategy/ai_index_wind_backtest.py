"""
AI Index Backtest using Wind API
================================
Creates an equal-weighted index of top 10 AI-related stocks in China A-share market,
fetches real historical data from Wind terminal, and performs backtest analysis.

Requirements:
- Windows OS (Wind terminal only runs on Windows)
- Wind terminal must be running and logged in
- Active Wind subscription with data access
- WindPy installed via Wind terminal's "Plugin Repair" feature

Usage:
    python example/ai_index_wind_backtest.py
"""

import sys

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("=" * 70)
    print("ERROR: WindPy module not found")
    print("=" * 70)
    print("\nWind API Requirements:")
    print("  1. Wind terminal only runs on Windows")
    print("  2. Open Wind terminal and go to 'Quantitative' section")
    print("  3. Select 'Plugin Repair' (插件修复) to install WindPy")
    print("  4. Run this script from the same Windows machine")
    print("\nCurrent platform:", sys.platform)
    print("=" * 70)
    sys.exit(1)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# Configuration
# ============================================================

# Top 10 AI-related stocks in China A-share market (as of 2024)
# Selected based on AI/semiconductor/computing exposure
AI_STOCKS = {
    "002415.SZ": "海康威视",  # Hikvision - AI vision
    "300496.SZ": "中科创达",  # ThunderSoft - AI OS
    "688111.SH": "金山办公",  # Kingsoft Office - AI office
    "688041.SH": "海光信息",  # Hygon - AI chips
    "688256.SH": "寒武纪",    # Cambricon - AI chips
    "300124.SZ": "汇川技术",  # Inovance - Industrial AI
    "002230.SZ": "科大讯飞",  # iFlytek - AI voice
    "688396.SH": "华润微",    # CR Micro - AI semiconductors
    "603501.SH": "韦尔股份",  # Will Semiconductor - AI image sensors
    "688981.SH": "中芯国际",  # SMIC - AI chip manufacturing
}

BACKTEST_START = "2023-01-01"
BACKTEST_END = "2025-01-20"

# ============================================================
# Wind Connection
# ============================================================

def start_wind_connection():
    """Start Wind connection with error handling."""
    print("Connecting to Wind terminal...")
    result = w.start(waitTime=60)

    if not w.isconnected():
        raise ConnectionError(
            "Failed to connect to Wind terminal. "
            "Please ensure Wind terminal is running and logged in."
        )
    print("Wind connection established successfully.")
    return True


def fetch_wind_data(func, *args, **kwargs):
    """Wrapper for Wind API calls with error handling."""
    error_code, result = func(*args, **kwargs)

    if error_code != 0:
        error_messages = {
            -40520007: "No data access permission",
            -40520010: "Request timeout",
            -40521001: "Invalid parameter",
            -40521003: "Invalid date range",
            -40522001: "Data not available",
            -40520004: "Network error",
            -40520005: "Server busy",
        }
        msg = error_messages.get(error_code, f"Unknown error code: {error_code}")
        raise ValueError(f"Wind API Error ({error_code}): {msg}")

    return result


# ============================================================
# Data Fetching
# ============================================================

def fetch_stock_prices(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices for multiple stocks.

    Returns:
        DataFrame with dates as index and stock codes as columns
    """
    symbols_str = ",".join(symbols)
    print(f"Fetching price data for {len(symbols)} stocks...")
    print(f"  Period: {start_date} to {end_date}")

    df = fetch_wind_data(
        w.wsd,
        symbols_str,
        "close",
        start_date,
        end_date,
        "PriceAdj=F;Fill=Previous",  # Forward-adjusted prices, fill missing
        usedf=True
    )

    # Rename columns to stock codes
    if len(symbols) == 1:
        df.columns = symbols

    print(f"  Retrieved {len(df)} trading days")
    return df


def fetch_stock_info(symbols: list) -> pd.DataFrame:
    """Fetch stock information (name, sector, market cap)."""
    symbols_str = ",".join(symbols)
    print("Fetching stock information...")

    df = fetch_wind_data(
        w.wss,
        symbols_str,
        "sec_name,industry_sw,mkt_cap_ard",
        usedf=True
    )

    df.columns = ["Name", "Industry", "MarketCap"]
    return df


def fetch_benchmark_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch CSI 300 index as benchmark."""
    print("Fetching CSI 300 benchmark data...")

    df = fetch_wind_data(
        w.wsd,
        "000300.SH",
        "close",
        start_date,
        end_date,
        "Fill=Previous",
        usedf=True
    )

    df.columns = ["CSI300"]
    return df


# ============================================================
# Index Construction
# ============================================================

def construct_equal_weight_index(prices: pd.DataFrame) -> pd.Series:
    """
    Construct an equal-weighted index from constituent prices.

    Method: Daily rebalance to equal weights (simplified)
    """
    # Ensure datetime index
    prices.index = pd.to_datetime(prices.index)

    # Calculate daily returns
    returns = prices.pct_change()

    # Equal-weighted portfolio returns
    portfolio_returns = returns.mean(axis=1)

    # Construct index (base = 1000)
    index_values = (1 + portfolio_returns).cumprod() * 1000
    index_values.iloc[0] = 1000  # Set initial value
    index_values.name = "AI_Index"

    return index_values


# ============================================================
# Performance Analysis
# ============================================================

def calculate_performance_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
    """Calculate comprehensive performance metrics."""

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    trading_days = len(returns)
    years = trading_days / 252

    # Annualized return
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Volatility
    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(252)

    # Sharpe ratio (assuming 2.5% risk-free rate)
    rf_rate = 0.025
    sharpe_ratio = (annualized_return - rf_rate) / annualized_vol if annualized_vol > 0 else 0

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = cumulative / running_max - 1
    max_drawdown = drawdowns.min()

    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - rf_rate) / downside_vol if downside_vol > 0 else 0

    metrics = {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_vol:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Sortino Ratio": f"{sortino_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Calmar Ratio": f"{calmar_ratio:.2f}",
        "Trading Days": trading_days,
    }

    # Benchmark comparison
    if benchmark_returns is not None:
        bench_total = (1 + benchmark_returns).prod() - 1
        bench_ann_return = (1 + bench_total) ** (1 / years) - 1
        excess_return = annualized_return - bench_ann_return

        # Beta and Alpha
        cov_matrix = np.cov(returns.dropna(), benchmark_returns.dropna())
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
        alpha = annualized_return - (rf_rate + beta * (bench_ann_return - rf_rate))

        # Information ratio
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
        info_ratio = excess_return / tracking_error if tracking_error > 0 else 0

        metrics.update({
            "Benchmark Return": f"{bench_ann_return:.2%}",
            "Excess Return": f"{excess_return:.2%}",
            "Beta": f"{beta:.2f}",
            "Alpha": f"{alpha:.2%}",
            "Information Ratio": f"{info_ratio:.2f}",
            "Tracking Error": f"{tracking_error:.2%}",
        })

    return metrics


def calculate_monthly_returns(index_values: pd.Series) -> pd.DataFrame:
    """Calculate monthly return table."""
    # Ensure datetime index
    index_values = index_values.copy()
    index_values.index = pd.to_datetime(index_values.index)

    monthly = index_values.resample('ME').last()
    monthly_returns = monthly.pct_change()

    # Create pivot table (Year x Month)
    df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })

    pivot = df.pivot(index='Year', columns='Month', values='Return')

    # Rename columns based on what's present
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[m-1] for m in pivot.columns]

    # Add yearly total
    yearly = index_values.resample('YE').last().pct_change()
    yearly.index = yearly.index.year
    pivot['Year Total'] = yearly

    return pivot


# ============================================================
# Reporting
# ============================================================

def print_report(
    stock_info: pd.DataFrame,
    index_values: pd.Series,
    benchmark_values: pd.Series,
    metrics: dict,
    monthly_returns: pd.DataFrame
):
    """Print comprehensive backtest report."""

    print("\n" + "=" * 70)
    print("AI INDEX BACKTEST REPORT")
    print("=" * 70)

    print("\n### Index Constituents ###")
    print(stock_info.to_string())

    print("\n### Performance Metrics ###")
    for key, value in metrics.items():
        print(f"  {key:25s}: {value}")

    print("\n### Monthly Returns ###")
    # Format as percentages
    formatted = monthly_returns.map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    print(formatted.to_string())

    print("\n### Index Summary ###")
    print(f"  Start Date: {index_values.index[0].strftime('%Y-%m-%d')}")
    print(f"  End Date: {index_values.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Initial Value: {index_values.iloc[0]:.2f}")
    print(f"  Final Value: {index_values.iloc[-1]:.2f}")

    print("\n" + "=" * 70)


# ============================================================
# Main Execution
# ============================================================

def main():
    """Run the AI index backtest."""

    try:
        # Start Wind connection
        start_wind_connection()

        # Get stock symbols
        symbols = list(AI_STOCKS.keys())

        # Fetch data
        stock_info = fetch_stock_info(symbols)
        stock_prices = fetch_stock_prices(symbols, BACKTEST_START, BACKTEST_END)
        benchmark_prices = fetch_benchmark_prices(BACKTEST_START, BACKTEST_END)

        # Validate data - no fallback to simulated data
        if stock_prices.empty:
            raise ValueError("No stock price data retrieved from Wind. Cannot proceed without real data.")

        missing_stocks = stock_prices.isnull().all()
        if missing_stocks.any():
            missing = [s for s, m in zip(symbols, missing_stocks) if m]
            raise ValueError(f"Missing data for stocks: {missing}. Cannot proceed without complete data.")

        print(f"\nData validation passed: {len(symbols)} stocks with complete price history")

        # Construct index
        print("\nConstructing equal-weighted AI index...")
        ai_index = construct_equal_weight_index(stock_prices)

        # Calculate returns
        index_returns = ai_index.pct_change().dropna()
        benchmark_returns = benchmark_prices['CSI300'].pct_change().dropna()

        # Align data
        common_dates = index_returns.index.intersection(benchmark_returns.index)
        index_returns = index_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]

        # Calculate metrics
        metrics = calculate_performance_metrics(index_returns, benchmark_returns)

        # Monthly returns
        monthly_returns = calculate_monthly_returns(ai_index)

        # Print report
        print_report(
            stock_info,
            ai_index,
            benchmark_prices['CSI300'],
            metrics,
            monthly_returns
        )

        # Return data for further analysis
        return {
            'index': ai_index,
            'benchmark': benchmark_prices['CSI300'],
            'stock_prices': stock_prices,
            'metrics': metrics,
            'monthly_returns': monthly_returns
        }

    except ConnectionError as e:
        print(f"\nConnection Error: {e}")
        print("\nTo run this backtest:")
        print("1. Ensure Wind terminal is running on your Windows machine")
        print("2. Log in to your Wind account")
        print("3. Run this script from the same machine")
        raise

    except ValueError as e:
        print(f"\nData Error: {e}")
        print("\nThis script requires real Wind data and does not support simulated data fallback.")
        raise

    finally:
        # Always close Wind connection
        if w.isconnected():
            w.stop()
            print("\nWind connection closed.")


if __name__ == "__main__":
    results = main()
