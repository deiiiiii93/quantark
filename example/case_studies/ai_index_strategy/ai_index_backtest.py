"""
AI Index Backtest - Top 10 AI Stocks in China A-Share Market

This script creates a custom index from top 10 AI-related stocks in China A-share market
and backtests its performance.

Uses AKShare's Sina data source (stock_zh_a_daily) which is more reliable than EastMoney.
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta


def fetch_with_retry(func, max_retries=3, delay=2, **kwargs):
    """Fetch data with retry logic to handle rate limiting."""
    for attempt in range(max_retries):
        try:
            result = func(**kwargs)
            return result
        except Exception as e:
            print(f" (retry {attempt + 1}: {type(e).__name__})", end="")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


# Top 10 AI stocks in China A-share market
# Format: (sina_symbol, code, name, description)
# Sina symbols need exchange prefix: sz for Shenzhen (000/002/300), sh for Shanghai (600/603/688)
TOP_AI_STOCKS = [
    ("sz002230", "002230", "科大讯飞", "AI语音龙头"),     # iFlytek - AI voice/NLP leader
    ("sz300496", "300496", "中科创达", "智能OS/AIoT"),   # ThunderSoft - Smart OS, AIoT
    ("sh688111", "688111", "金山办公", "AI办公应用"),    # Kingsoft Office - AI office apps
    ("sz300033", "300033", "同花顺", "AI金融科技"),      # Hithink - AI fintech
    ("sh603019", "603019", "中科曙光", "AI算力服务器"),  # Sugon - AI computing servers
    ("sh688256", "688256", "寒武纪", "AI芯片"),          # Cambricon - AI chips (STAR)
    ("sz002415", "002415", "海康威视", "AI视觉安防"),    # Hikvision - AI vision/security
    ("sh688041", "688041", "海光信息", "国产AI芯片"),    # Hygon - Domestic AI chips
    ("sz300474", "300474", "景嘉微", "GPU芯片"),         # Jingjia Micro - GPU chips
    ("sh688561", "688561", "奇安信", "AI安全"),          # Qi An Xin - AI cybersecurity
]


def fetch_historical_data(symbols, start_date, end_date):
    """
    Fetch historical daily data using Sina source (stock_zh_a_daily).

    This is more reliable than EastMoney source (stock_zh_a_hist) which often
    gets blocked or rate-limited.
    """
    print(f"\nFetching historical data from {start_date} to {end_date}...")
    print("(Using Sina data source - stock_zh_a_daily)\n")

    all_data = {}
    failed = []

    for i, (sina_symbol, code, name, _) in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {code} {name}...", end="", flush=True)
        time.sleep(0.5)  # Brief delay to avoid rate limiting

        df = fetch_with_retry(
            ak.stock_zh_a_daily,
            symbol=sina_symbol,
            adjust="qfq"  # Forward-adjusted
        )

        if df is not None and not df.empty:
            # Sina returns 'date' column as string, convert to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Filter to date range
            df = df[(df.index >= pd.to_datetime(start_date, format='%Y%m%d')) &
                    (df.index <= pd.to_datetime(end_date, format='%Y%m%d'))]

            if not df.empty:
                all_data[code] = df['close']  # Sina uses English column names
                print(f" OK ({len(df)} days)")
            else:
                print(f" EMPTY (no data in range)")
                failed.append(code)
        else:
            print(" FAILED")
            failed.append(code)

    if failed:
        print(f"\nWarning: Failed to fetch data for: {failed}")

    # Combine into single DataFrame
    if all_data:
        prices_df = pd.DataFrame(all_data)
        prices_df = prices_df.sort_index()
        # Forward fill then backward fill for any gaps
        prices_df = prices_df.ffill().bfill()
    else:
        prices_df = pd.DataFrame()

    return prices_df


def calculate_equal_weight_index(prices_df):
    """Calculate equal-weighted index returns."""
    # Calculate daily returns
    returns = prices_df.pct_change()

    # Equal-weighted portfolio return (mean of all stock returns)
    portfolio_returns = returns.mean(axis=1)

    # Cumulative returns (index level starting at 100)
    index_level = (1 + portfolio_returns).cumprod() * 100
    index_level.iloc[0] = 100  # Start at 100

    return portfolio_returns, index_level


def calculate_performance_metrics(returns, index_level):
    """Calculate key performance metrics."""
    returns_clean = returns.dropna()

    if len(returns_clean) == 0:
        return {}

    # Total return
    total_return = (index_level.iloc[-1] / index_level.iloc[0] - 1) * 100

    # Annualized return (assuming 250 trading days)
    n_days = len(returns_clean)
    annualized_return = ((1 + total_return/100) ** (250/n_days) - 1) * 100 if n_days > 0 else 0

    # Volatility (annualized)
    volatility = returns_clean.std() * np.sqrt(250) * 100

    # Sharpe ratio (assuming 2.5% risk-free rate for China)
    rf_rate = 0.025
    sharpe = (annualized_return/100 - rf_rate) / (volatility/100) if volatility > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = returns_clean[returns_clean < 0]
    downside_std = downside_returns.std() * np.sqrt(250) if len(downside_returns) > 0 else 0
    sortino = (annualized_return/100 - rf_rate) / downside_std if downside_std > 0 else 0

    # Maximum drawdown
    rolling_max = index_level.expanding().max()
    drawdown = (index_level - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate = (returns_clean > 0).sum() / len(returns_clean) * 100

    # Best and worst days
    best_day = returns_clean.max() * 100
    worst_day = returns_clean.min() * 100

    # Average daily return
    avg_daily = returns_clean.mean() * 100

    # Skewness and Kurtosis
    skewness = returns_clean.skew()
    kurtosis = returns_clean.kurtosis()

    return {
        'Total Return (%)': total_return,
        'Annualized Return (%)': annualized_return,
        'Annualized Volatility (%)': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown (%)': max_drawdown,
        'Win Rate (%)': win_rate,
        'Avg Daily Return (%)': avg_daily,
        'Best Day (%)': best_day,
        'Worst Day (%)': worst_day,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Trading Days': n_days
    }


def fetch_benchmark_data(start_date, end_date):
    """Fetch benchmark index data using Sina source (stock_zh_index_daily)."""
    benchmarks = {}

    # Index symbols for Sina source (stock_zh_index_daily)
    benchmark_list = [
        ("sh000300", "沪深300"),   # CSI 300
        ("sz399006", "创业板指"),  # ChiNext
    ]

    start_dt = pd.to_datetime(start_date, format='%Y%m%d')
    end_dt = pd.to_datetime(end_date, format='%Y%m%d')

    for idx_symbol, idx_name in benchmark_list:
        print(f"  Fetching {idx_name} ({idx_symbol})...", end="", flush=True)
        time.sleep(0.5)

        try:
            # Use stock_zh_index_daily (Sina source) - more reliable than index_zh_a_hist
            df = fetch_with_retry(
                ak.stock_zh_index_daily,
                symbol=idx_symbol
            )

            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

                # Filter to date range
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]

                if not df.empty:
                    benchmarks[idx_name] = df['close'].pct_change()
                    print(f" OK ({len(df)} days)")
                else:
                    print(" EMPTY (no data in range)")
            else:
                print(" FAILED")
        except Exception as e:
            print(f" FAILED ({type(e).__name__})")

    return benchmarks


def compare_with_benchmark(index_returns, benchmarks):
    """Compare portfolio with benchmark indices."""
    if not benchmarks:
        return None

    results = {}
    index_returns_clean = index_returns.dropna()

    for bench_name, bench_returns in benchmarks.items():
        # Align dates
        common_dates = index_returns_clean.index.intersection(bench_returns.index)

        if len(common_dates) < 10:
            continue

        idx_ret = index_returns_clean.loc[common_dates]
        bench_ret = bench_returns.loc[common_dates]

        # Calculate beta
        covariance = idx_ret.cov(bench_ret)
        variance = bench_ret.var()
        beta = covariance / variance if variance > 0 else 0

        # Calculate alpha (annualized)
        rf_daily = 0.025 / 250
        alpha = (idx_ret.mean() - rf_daily - beta * (bench_ret.mean() - rf_daily)) * 250 * 100

        # Correlation
        correlation = idx_ret.corr(bench_ret)

        # Information ratio
        tracking_error = (idx_ret - bench_ret).std() * np.sqrt(250)
        excess_return_annual = (idx_ret.mean() - bench_ret.mean()) * 250
        info_ratio = excess_return_annual / tracking_error if tracking_error > 0 else 0

        # Benchmark total return
        bench_cumret = (1 + bench_ret).cumprod()
        bench_total_return = (bench_cumret.iloc[-1] - 1) * 100

        results[bench_name] = {
            'Beta': beta,
            'Alpha (%)': alpha,
            'Correlation': correlation,
            'Information Ratio': info_ratio,
            'Benchmark Return (%)': bench_total_return
        }

    return results


def main():
    """Main function to run the AI index backtest."""
    print("=" * 70)
    print("     AI INDEX BACKTEST - Top 10 AI Stocks in China A-Share")
    print("=" * 70)

    # Set date range (past 1 year)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    print(f"\nBacktest Period: {start_date} to {end_date}")

    # Display selected stocks
    print("\n" + "-" * 70)
    print("SELECTED AI STOCKS")
    print("-" * 70)
    for i, (_, code, name, desc) in enumerate(TOP_AI_STOCKS, 1):
        print(f"  {i:2d}. {code} {name:10s} - {desc}")

    # Fetch historical data
    prices_df = fetch_historical_data(TOP_AI_STOCKS, start_date, end_date)

    if prices_df.empty:
        print("\nError: No historical data available.")
        return None

    print(f"\nData retrieved for {len(prices_df.columns)} stocks, {len(prices_df)} trading days")

    # Drop stocks with too much missing data (>20%)
    valid_threshold = len(prices_df) * 0.8
    valid_stocks = prices_df.columns[prices_df.notna().sum() > valid_threshold].tolist()

    if len(valid_stocks) < len(prices_df.columns):
        dropped = set(prices_df.columns) - set(valid_stocks)
        print(f"Note: Dropped {dropped} due to insufficient data")
        prices_df = prices_df[valid_stocks]

    # Calculate equal-weighted index
    portfolio_returns, index_level = calculate_equal_weight_index(prices_df)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_returns, index_level)

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    for key, value in metrics.items():
        if key == 'Trading Days':
            print(f"  {key:30s}: {value:>12d}")
        else:
            print(f"  {key:30s}: {value:>12.2f}")

    # Fetch and compare with benchmarks
    print("\n" + "-" * 70)
    print("BENCHMARK COMPARISON")
    print("-" * 70)

    benchmarks = fetch_benchmark_data(start_date, end_date)
    benchmark_stats = compare_with_benchmark(portfolio_returns, benchmarks)

    if benchmark_stats:
        for bench_name, stats in benchmark_stats.items():
            print(f"\n  vs {bench_name}:")
            for key, value in stats.items():
                print(f"    {key:25s}: {value:>12.2f}")

            excess_return = metrics['Total Return (%)'] - stats['Benchmark Return (%)']
            print(f"    {'Excess Return (%)':25s}: {excess_return:>12.2f}")

    # Monthly returns breakdown
    print("\n" + "-" * 70)
    print("MONTHLY RETURNS")
    print("-" * 70)

    # Resample to month-end
    monthly_returns = portfolio_returns.resample('ME').apply(lambda x: (1+x).prod() - 1) * 100

    # Format monthly returns
    for date_idx in monthly_returns.index:
        ret = monthly_returns.loc[date_idx]
        month_str = date_idx.strftime('%Y-%m')
        bar_len = min(int(abs(ret) / 2), 30)  # Cap bar length
        bar = "█" * bar_len if ret > 0 else "▒" * bar_len
        sign = "+" if ret > 0 else ""
        print(f"  {month_str}: {sign}{ret:>7.2f}%  {bar}")

    # Individual stock performance
    print("\n" + "-" * 70)
    print("INDIVIDUAL STOCK RETURNS (Period)")
    print("-" * 70)

    stock_returns = {}
    for col in prices_df.columns:
        first_valid = prices_df[col].first_valid_index()
        last_valid = prices_df[col].last_valid_index()
        if first_valid is not None and last_valid is not None:
            ret = (prices_df[col].loc[last_valid] / prices_df[col].loc[first_valid] - 1) * 100
            stock_returns[col] = ret

    # Sort by return descending
    sorted_returns = sorted(stock_returns.items(), key=lambda x: x[1], reverse=True)

    # Find stock name from our list
    stock_map = {code: name for _, code, name, _ in TOP_AI_STOCKS}

    for code, ret in sorted_returns:
        name = stock_map.get(code, code)
        sign = "+" if ret > 0 else ""
        bar_len = min(int(abs(ret) / 5), 30)  # Cap bar length
        bar = "█" * bar_len if ret > 0 else "▒" * bar_len
        print(f"  {code} {name:10s}: {sign}{ret:>8.2f}%  {bar}")

    # Correlation matrix
    print("\n" + "-" * 70)
    print("STOCK CORRELATION MATRIX")
    print("-" * 70)

    returns_df = prices_df.pct_change().dropna()
    corr_matrix = returns_df.corr()

    # Print header with shorter codes
    cols = list(corr_matrix.columns)
    header = "          " + "  ".join([c[-4:] for c in cols])
    print(header)

    for i, row_code in enumerate(cols):
        row_str = f"  {row_code[-4:]}    "
        for j, col_code in enumerate(cols):
            val = corr_matrix.loc[row_code, col_code]
            if i == j:
                row_str += " 1.00"
            else:
                row_str += f" {val:.2f}"
        print(row_str)

    # Risk contribution analysis
    print("\n" + "-" * 70)
    print("RISK CONTRIBUTION")
    print("-" * 70)

    # Calculate marginal risk contribution
    cov_matrix = returns_df.cov() * 250  # Annualized
    weights = np.ones(len(cols)) / len(cols)  # Equal weights
    port_var = weights @ cov_matrix.values @ weights
    port_vol = np.sqrt(port_var)

    # Marginal contribution to risk
    mcr = (cov_matrix.values @ weights) / port_vol
    # Component contribution to risk
    ccr = weights * mcr
    # Percentage contribution
    pcr = ccr / port_vol * 100

    for i, code in enumerate(cols):
        name = stock_map.get(code, code)
        print(f"  {code} {name:10s}: {pcr[i]:>6.2f}% of portfolio risk")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return {
        'stocks': TOP_AI_STOCKS,
        'prices': prices_df,
        'returns': portfolio_returns,
        'index_level': index_level,
        'metrics': metrics,
        'benchmark_stats': benchmark_stats
    }


if __name__ == "__main__":
    results = main()
